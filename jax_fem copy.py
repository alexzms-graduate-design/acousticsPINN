# jax_fem.py
import jax
import jax.numpy as jnp
from jax import jit, grad, vmap
from functools import partial
import numpy as np
import meshio
from tqdm import tqdm
import pyvista as pv
import matplotlib.pyplot as plt
import scipy.sparse as sps
import matplotlib.colors as mcolors # For LogNorm

# 管道参数定义
r_outer = 0.05   # 外半径 5cm
r_inner = 0.045  # 内半径 4.5cm
angle_deg = 45     # 分支角度 (degrees)
length_main = 1.5  # 主干总长 1.5m
x_junction = 0.5   # Junction point x-coordinate
# Branch length MUST match the value used in geometry_gmsh.py
length_branch = 0.3 

# Add rho_f near the top definitions or pass it
rho_f = 1.225 # kg/m^3 (Air density)

# ===============================
# 辅助函数：线性四面体元的计算
# ===============================
@jit
def compute_tetra_volume(coords):
    """
    计算四面体体积
    coords: [4,3] 每行为节点坐标
    Volume = abs(det([x2-x1, x3-x1, x4-x1]))/6
    """
    v0 = coords[1] - coords[0]
    v1 = coords[2] - coords[0]
    v2 = coords[3] - coords[0]
    vol = jnp.abs(jnp.linalg.det(jnp.stack([v0, v1, v2], axis=1))) / 6.0
    return vol

@jit
def compute_shape_function_gradients(coords):
    """
    计算四面体线性形函数梯度（常数）。
    方法：对齐次线性系统 [1 x y z] 的逆矩阵的后三行。
    返回: [4,3]，每一行为对应节点形函数在 x,y,z 方向的梯度。
    """
    ones = jnp.ones((4,1), dtype=coords.dtype)
    A = jnp.concatenate([ones, coords], axis=1)  # [4,4]
    A_inv = jnp.linalg.inv(A)
    # 后三行为 b, c, d 系数，对应梯度
    grads = jnp.transpose(A_inv[1:,:])   # shape [4,3]
    return grads

@jit
def element_matrices_fluid(coords):
    """
    计算流体四面体单元的局部刚度和质量矩阵。
    使用公式：
      K_e = Volume * (grad(N) @ grad(N)^T)
      M_e = Volume/20 * (I + ones(4,4))
    """
    V = compute_tetra_volume(coords)
    grads = compute_shape_function_gradients(coords)  # [4,3]
    K_e = V * (grads @ jnp.transpose(grads))
    ones_4 = jnp.ones((4,4), dtype=coords.dtype)
    M_e = V / 20.0 * (jnp.eye(4, dtype=coords.dtype) + ones_4)
    return K_e, M_e

@jit
def elasticity_D_matrix(E, nu):
    """
    3D各向同性弹性材料 D 矩阵 (6x6)，Voigt 表示。
    """
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    D = coeff * jnp.array([
        [1 - nu,    nu,    nu,       0,       0,       0],
        [nu,    1 - nu,    nu,       0,       0,       0],
        [nu,      nu,  1 - nu,       0,       0,       0],
        [0,       0,      0,  (1 - 2 * nu) / 2,  0,       0],
        [0,       0,      0,       0,  (1 - 2 * nu) / 2,  0],
        [0,       0,      0,       0,       0,  (1 - 2 * nu) / 2]
    ], dtype=jnp.float32)
    return D

@jit
def compute_B_matrix(coords):
    """
    构造四面体单元的应变-位移矩阵 B (6 x 12)。
    利用线性形函数梯度计算，各节点 3 dof。
    """
    grads = compute_shape_function_gradients(coords)  # [4,3]
    
    # JAX doesn't allow in-place updates, so we'll build each row separately
    # and then stack them together
    rows = []
    for i in range(4):
        bx, by, bz = grads[i]
        row_block = jnp.array([
            [bx,    0,    0],
            [0,   by,     0],
            [0,    0,    bz],
            [by,   bx,    0],
            [0,    bz,   by],
            [bz,   0,    bx]
        ], dtype=coords.dtype)
        
        # Create a row with zeros in the right places
        row = jnp.zeros((6, 12), dtype=coords.dtype)
        row = row.at[:, i*3:(i+1)*3].set(row_block)
        rows.append(row)
    
    # Sum all rows to get the final B matrix
    B = sum(rows)
    return B

@jit
def element_matrices_solid(coords, E, nu, rho_s):
    """
    计算固体四面体单元（管壁）的局部刚度和质量矩阵。
    结构单元 DOF = 3 per node, 整体矩阵尺寸 12x12。
    """
    V = compute_tetra_volume(coords)
    B = compute_B_matrix(coords)  # [6, 12]
    D = elasticity_D_matrix(E, nu)  # [6,6]
    # 刚度矩阵：K_e = V * B^T D B
    K_e = V * (jnp.transpose(B) @ (D @ B))
    
    # 质量矩阵采用对角 lumped mass:
    # In JAX we need to construct this differently since we can't do in-place updates
    m_lump = rho_s * V / 4.0
    diag_blocks = []
    for i in range(4):
        diag_blocks.append((i*3, i*3+3, m_lump * jnp.eye(3, dtype=coords.dtype)))
    
    # Now we need to build a matrix from these blocks
    M_e = jnp.zeros((12, 12), dtype=coords.dtype)
    for start_idx, end_idx, block in diag_blocks:
        M_e = M_e.at[start_idx:end_idx, start_idx:end_idx].set(block)
    
    return K_e, M_e

def visualize_system(A, f, n_nodes, n_solid_dof, title_suffix="Raw System"):
    """
    Visualizes the matrix A using colored scatter plot (log scale) and vector f as a heatmap.
    Marks the divisions between fluid and solid DOFs.
    """
    print(f"[Visualizing System] 矩阵形状: {A.shape}, 向量形状: {f.shape}")
    print(f"[Visualizing System] 流体自由度(n_nodes): {n_nodes}, 固体自由度: {n_solid_dof}")
    total_dof = n_nodes + n_solid_dof
    if A.shape[0] != total_dof or A.shape[1] != total_dof or f.shape[0] != total_dof:
        print(f"[Warning] visualize_system中的维度不匹配: A={A.shape}, f={f.shape}, 预期总自由度={total_dof}")
        actual_total_dof = A.shape[0]
        if n_nodes > actual_total_dof: n_nodes = actual_total_dof
    else:
        actual_total_dof = total_dof

    # Move data to CPU
    A_cpu = np.array(A)
    f_cpu = np.array(f)

    # Convert dense numpy array to scipy sparse COO matrix for easier access to values/coords
    A_sparse_coo = sps.coo_matrix(A_cpu)
    
    # Filter out small values based on tolerance
    tol = 1e-9
    mask = np.abs(A_sparse_coo.data) > tol
    row = A_sparse_coo.row[mask]
    col = A_sparse_coo.col[mask]
    data = A_sparse_coo.data[mask]
    
    num_non_zero = len(data)
    print(f"[Visualizing System] 矩阵稀疏度: {num_non_zero / (A.shape[0] * A.shape[1]) * 100:.4f}% 非零元素 > {tol}.")
    
    if num_non_zero == 0:
        print("[Warning] 矩阵A中未找到非零元素(高于阈值)，跳过绘图.")
        return

    # Adjust subplot layout: give more width to matrix plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [3, 1]}) 
    fig.suptitle(f"System Visualization ({title_suffix})")

    # --- Matrix Plot (Colored Scatter) ---
    ax = axs[0]
    # Use absolute values for color, apply LogNorm if data range is large
    abs_data = np.abs(data)
    min_val, max_val = abs_data.min(), abs_data.max()
    if min_val <= 0 or max_val / min_val < 100: # Use linear scale if min is zero/negative or range is small
        norm = mcolors.Normalize(vmin=min_val, vmax=max_val)
    else:
        norm = mcolors.LogNorm(vmin=min_val, vmax=max_val)
        
    cmap = plt.get_cmap('viridis') # Choose a colormap
    
    # Scatter plot: row index vs column index, colored by value magnitude
    # Invert row axis to match matrix layout (optional)
    scatter = ax.scatter(col, row, c=abs_data, cmap=cmap, norm=norm, s=0.5, marker='.') # Use small dots
    ax.set_title(f"Matrix A ({A.shape[0]}x{A.shape[1]}) Non-Zero Elements (Log Color Scale)")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    ax.set_xlim(-0.5, A.shape[1]-0.5) # Adjust xlim slightly for text
    ax.set_ylim(A.shape[0]-0.5, -0.5) # Adjust ylim slightly for text
    ax.set_aspect('equal')
    ax.grid(True, linestyle=':', linewidth=0.5)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Absolute Value of Elements')

    # Add lines and text to mark blocks
    if n_nodes < actual_total_dof:
        # Lines separating blocks
        ax.axvline(n_nodes - 0.5, color='r', linestyle='--', linewidth=1)
        ax.axhline(n_nodes - 0.5, color='r', linestyle='--', linewidth=1)

        # Text labels for axis regions
        ax.text((n_nodes / 2) / actual_total_dof, 1.02, 'Fluid DOFs', ha='center', va='bottom', transform=ax.transAxes, color='red', fontsize=9)
        ax.text((n_nodes + n_solid_dof / 2) / actual_total_dof, 1.02, 'Solid DOFs', ha='center', va='bottom', transform=ax.transAxes, color='red', fontsize=9)
        ax.text(-0.02, (n_nodes / 2) / actual_total_dof, 'Fluid DOFs', rotation=90, ha='right', va='center', transform=ax.transAxes, color='red', fontsize=9)
        ax.text(-0.02, (n_nodes + n_solid_dof / 2) / actual_total_dof, 'Solid DOFs', rotation=90, ha='right', va='center', transform=ax.transAxes, color='red', fontsize=9)

    # --- Force Vector Plot (Heatmap) ---
    ax = axs[1]
    # Reshape f_cpu to (N, 1) for matshow
    f_plot = f_cpu.reshape(-1, 1)
    img = ax.matshow(f_plot, cmap='coolwarm', aspect='auto') # Use 'auto' aspect for vector
    ax.set_title(f"Vector f ({f.shape[0]}x1)")
    ax.set_xlabel("Column")
    ax.set_ylabel("Index")
    # Disable x-axis ticks/labels as it's just one column
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    ax.grid(True, which='major', axis='y', linestyle='-', linewidth=0.5)

    # Add colorbar for vector
    cbar_f = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar_f.set_label('Value')

    # Add line to separate fluid/solid parts
    if n_nodes < actual_total_dof:
        # Draw horizontal line on heatmap
        ax.axhline(n_nodes - 0.5, color='r', linestyle='--', linewidth=1)
        # Add text annotations for fluid/solid regions
        ax.text(0.5, n_nodes / 2, 'Fluid', rotation=90, va='center', ha='center', color='red', transform=ax.transAxes, fontsize=8)
        ax.text(0.5, (n_nodes + actual_total_dof) / 2 / actual_total_dof, 'Solid', rotation=90, va='center', ha='center', color='red', transform=ax.transAxes, fontsize=8)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save_path = f'system_visualization_{title_suffix.replace(" ", "_")}.png'
    plt.savefig(save_path, dpi=150)
    print(f"[Visualizing System] 图表已保存到 {save_path}")
    plt.close(fig)

# ===============================
# Coupled FEM 求解器（全局组装、求解及后处理）
# ===============================
class CoupledFEMSolver:
    def __init__(self, mesh_file, frequency=1000.0, cppenalty=1e8, bcpenalty=1e6):
        """
        mesh_file: "y_pipe.msh" 路径
        frequency: 噪音源频率 (Hz)
        penalty: 流固耦合惩罚参数 β
        """
        self.freq = frequency
        self.omega = 2 * np.pi * frequency
        self.c_f = 343.0
        self.cppenalty = cppenalty
        self.bcpenalty = bcpenalty
        
        # 读网格文件（使用 meshio ）
        print(f"[info] 正在读取网格文件: {mesh_file}")
        mesh = meshio.read(mesh_file)
        # 提取节点（3D 坐标）
        self.nodes = jnp.array(mesh.points[:, :3], dtype=jnp.float32)
        
        # 检查是否存在四面体单元和物理标签
        if not hasattr(mesh, 'cells'):
            raise ValueError("Mesh does not contain cell blocks.")

        # --- 提取物理组名称和标签映射 ---
        tag_names = {}
        fluid_tag = solid_tag = interface_tag = None
        
        if hasattr(mesh, 'field_data'):
            print("[info] 从field_data提取物理组标签映射...")
            for name, data in mesh.field_data.items():
                if len(data) >= 2:
                    tag = data[0]
                    tag_names[tag] = name
                    print(f"物理组: {name} -> 标签: {tag}")
                    if name.lower() == 'fluid':
                        fluid_tag = tag
                    elif name.lower() == 'wall' or name.lower() == 'solid':
                        solid_tag = tag
                    elif 'interface' in name.lower() or 'fluidsolid' in name.lower().replace(" ", ""):
                        interface_tag = tag
                        print(f"[info] 找到流固界面物理组: {name}, 标签: {tag}")
        
        if fluid_tag is None or solid_tag is None:
            print("[warning] 未能从field_data自动识别流体和固体标签，尝试使用默认标签...")
            fluid_tag = 1
            solid_tag = 2
        
        print(f"[info] 使用流体标签: {fluid_tag}, 固体标签: {solid_tag}, 界面标签: {interface_tag}")

        # --- 提取流体和固体单元 ---
        fluid_elems_np = []
        solid_elems_np = []
        interface_elems_np = []
        
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "tetra":
                # 尝试找到对应的物理组标签
                if hasattr(mesh, 'cell_data') and 'gmsh:physical' in mesh.cell_data:
                    if i < len(mesh.cell_data['gmsh:physical']):
                        tags = np.unique(mesh.cell_data['gmsh:physical'][i])
                        if len(tags) == 1:  # 确保只有一个标签
                            tag = tags[0]
                            if tag == fluid_tag:
                                fluid_elems_np.append(cell_block.data)
                                print(f"找到流体块 {i}，包含 {cell_block.data.shape[0]} 个四面体单元")
                            elif tag == solid_tag:
                                solid_elems_np.append(cell_block.data)
                                print(f"找到固体块 {i}，包含 {cell_block.data.shape[0]} 个四面体单元")
                            elif tag == interface_tag:
                                interface_elems_np.append(cell_block.data)
                                print(f"找到界面块 {i}，包含 {cell_block.data.shape[0]} 个四面体单元")
            elif cell_block.type == "triangle":
                # 检查三角形单元是否属于界面
                if hasattr(mesh, 'cell_data') and 'gmsh:physical' in mesh.cell_data:
                    if i < len(mesh.cell_data['gmsh:physical']):
                        tags = np.unique(mesh.cell_data['gmsh:physical'][i])
                        if len(tags) == 1 and tags[0] == interface_tag:  
                            print(f"找到界面三角面片块 {i}，包含 {cell_block.data.shape[0]} 个三角形单元")
                            # 提取三角形单元的节点
                            interface_tri_nodes = cell_block.data
                            # 这些节点将在后面用于界面检测
        
        # 合并所有流体和固体单元
        if fluid_elems_np:
            fluid_elems_np = np.vstack(fluid_elems_np)
        else:
            fluid_elems_np = np.array([], dtype=np.int64).reshape(0, 4)
            print("[warning] 未找到流体单元!")
        
        if solid_elems_np:
            solid_elems_np = np.vstack(solid_elems_np)
        else:
            solid_elems_np = np.array([], dtype=np.int64).reshape(0, 4)
            print("[warning] 未找到固体单元!")
        
        self.fluid_elements = jnp.array(fluid_elems_np, dtype=jnp.int32)
        self.solid_elements = jnp.array(solid_elems_np, dtype=jnp.int32)

        print(f"[info] 加载了 {self.fluid_elements.shape[0]} 个流体单元 (tag {fluid_tag}) 和 {self.solid_elements.shape[0]} 个固体单元 (tag {solid_tag}).")

        # --- 创建节点映射 ---
        # Since we can't directly manipulate JAX arrays, use numpy temporarily
        fluid_elements_np = np.array(self.fluid_elements)
        solid_elements_np = np.array(self.solid_elements)
        
        fluid_unique_nodes_np = np.unique(fluid_elements_np.flatten())
        solid_unique_nodes_np = np.unique(solid_elements_np.flatten())
        
        self.fluid_unique_nodes = jnp.array(fluid_unique_nodes_np, dtype=jnp.int32)
        self.solid_unique_nodes = jnp.array(solid_unique_nodes_np, dtype=jnp.int32)
        
        self.N_fluid_unique = len(self.fluid_unique_nodes)
        self.N_solid_unique = len(self.solid_unique_nodes)
        self.n_solid_dof = self.N_solid_unique * 3

        # 全局索引 -> 局部索引映射
        self.fluid_mapping = {int(global_idx): local_idx for local_idx, global_idx in enumerate(fluid_unique_nodes_np)}
        self.solid_mapping = {int(global_idx): local_idx for local_idx, global_idx in enumerate(solid_unique_nodes_np)}

        print(f"[info] 找到 {self.N_fluid_unique} 个唯一流体节点和 {self.N_solid_unique} 个唯一固体节点 ({self.n_solid_dof} 个固体自由度).")
        
        # --- 识别流固界面节点 ---
        if interface_tag is not None:
            # 如果找到界面物理组，直接从gmsh中读取界面
            print("[info] 从gmsh物理组识别流固界面节点...")
            
            # 查找所有带有界面标签的单元
            interface_nodes_set = set()
            
            # 检查所有的网格单元块，寻找界面物理组
            for i, cell_block in enumerate(mesh.cells):
                if hasattr(mesh, 'cell_data') and 'gmsh:physical' in mesh.cell_data:
                    if i < len(mesh.cell_data['gmsh:physical']):
                        tags = np.unique(mesh.cell_data['gmsh:physical'][i])
                        if len(tags) == 1 and tags[0] == interface_tag:
                            # 添加此单元块中的所有节点到界面节点集合
                            for node_idx in cell_block.data.flatten():
                                interface_nodes_set.add(node_idx)
            
            # 转换为jax array
            self.interface_idx = jnp.array(list(interface_nodes_set), dtype=jnp.int32)
            print(f"[info] 从界面物理组找到 {len(interface_nodes_set)} 个流固界面节点")
        else:
            # 如果找不到界面物理组，退回到找流体和固体共享的节点
            print("[info] 未找到界面物理组，退回到通过流体和固体共享节点识别界面...")
            fluid_node_set = set(fluid_unique_nodes_np)
            solid_node_set = set(solid_unique_nodes_np)
            interface_node_set = fluid_node_set.intersection(solid_node_set)
            self.interface_idx = jnp.array(list(interface_node_set), dtype=jnp.int32)
            print(f"[info] 找到 {len(interface_node_set)} 个流固界面共享节点")

        # --- 计算界面法向量 ---
        print("[info] 计算界面法向量...")
        # Since we need to dynamically build a dictionary, we'll work with numpy arrays first
        nodes_np = np.array(self.nodes)
        interface_idx_np = np.array(self.interface_idx)
        fluid_elements_np = np.array(self.fluid_elements)
        
        node_to_face_normals = {node_id: [] for node_id in interface_idx_np}

        # 迭代流体单元找到界面面及其法向量
        for elem_nodes in tqdm(fluid_elements_np, desc="计算界面法向量"):
            nodes_coords = nodes_np[elem_nodes]  # 4个节点坐标
            # 定义四个面（局部索引）
            local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]

            for i_face, local_face in enumerate(local_faces):
                global_node_indices = elem_nodes[np.array(local_face)] # 面的全局节点索引

                # 检查是否所有3个节点都是界面节点
                is_interface_face = all(node_id in node_to_face_normals for node_id in global_node_indices)

                if is_interface_face:
                    # 计算面法向量
                    p0, p1, p2 = nodes_np[global_node_indices]
                    v1 = p1 - p0
                    v2 = p2 - p0
                    normal = np.cross(v1, v2)
                    norm_mag = np.linalg.norm(normal)

                    if norm_mag > 1e-12: # 避免除以零
                        normal = normal / norm_mag

                        # 确保法向量指向流体元素外部
                        local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                        p3 = nodes_coords[local_idx_p3]
                        face_centroid = (p0 + p1 + p2) / 3.0
                        vec_to_p3 = p3 - face_centroid

                        # 如果法向量指向p3（内部），翻转它
                        if np.dot(normal, vec_to_p3) > 0:
                            normal = -normal

                        # 将面法向量添加到其3个顶点的列表中
                        for node_id in global_node_indices:
                            if node_id in node_to_face_normals:
                                node_to_face_normals[node_id].append(normal)

        # 通过平均计算最终顶点法向量
        final_normals_list = []
        zero_normal_count = 0
        for node_id in tqdm(interface_idx_np, desc="平均节点法向量"):
            normals_to_average = node_to_face_normals.get(node_id, [])

            if not normals_to_average:
                avg_normal = np.zeros(3)
                zero_normal_count += 1
            else:
                avg_normal = np.sum(normals_to_average, axis=0)
                norm_val = np.linalg.norm(avg_normal)
                if norm_val < 1e-12:
                    avg_normal = np.zeros(3)
                    zero_normal_count += 1
                else:
                    avg_normal = avg_normal / norm_val
            final_normals_list.append(avg_normal)

        if zero_normal_count > 0:
            print(f"[warning] {zero_normal_count} 个界面节点的法向量为零.")
            
        self.interface_normals = jnp.array(final_normals_list, dtype=jnp.float32)
        
        # 过滤掉法向量为零的节点
        normal_magnitudes = jnp.linalg.norm(self.interface_normals, axis=1)
        norm_tolerance = 1e-9
        
        # In JAX, we need to be careful with filtering since arrays are immutable
        # We'll use numpy for the filtering then convert back to JAX arrays
        valid_normal_mask_np = np.array(normal_magnitudes) > norm_tolerance
        
        original_count = self.interface_idx.shape[0]
        self.interface_idx = jnp.array(interface_idx_np[valid_normal_mask_np], dtype=jnp.int32)
        self.interface_normals = jnp.array(np.array(self.interface_normals)[valid_normal_mask_np], dtype=jnp.float32)
        
        filtered_count = self.interface_idx.shape[0]
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"[info] 移除了 {removed_count} 个法向量无效的界面节点 (norm < {norm_tolerance}).")
        print(f"[info] 最终界面节点数量: {filtered_count}")
        
        # 创建界面节点集合，用于排除它们不被用作其他边界条件
        interface_idx_np = np.array(self.interface_idx)
        interface_nodes_set = set(interface_idx_np)

        # --- Inlet/Outlet Definitions ---
        # 设置入口条件（x ≈ 0）- 确保它们是流体节点，但不是界面节点
        # Inlet (x approx 0) - Ensure they are fluid nodes but not interface nodes
        nodes_np = np.array(self.nodes)
        potential_near_indices = np.nonzero(np.abs(nodes_np[:, 0]) < 1e-3)[0]
        near_mask = np.isin(potential_near_indices, fluid_unique_nodes_np)
        near_fluid_candidates = potential_near_indices[near_mask]
        # 排除界面节点
        non_interface_mask = np.array([idx not in interface_nodes_set for idx in near_fluid_candidates])
        self.near_fluid_idx = jnp.array(near_fluid_candidates[non_interface_mask], dtype=jnp.int32)
        print(f"[info] 识别入口节点: 总共 {near_fluid_candidates.shape[0]} 个候选节点, 排除 {near_fluid_candidates.shape[0] - len(non_interface_mask[non_interface_mask])} 个界面节点, 最终 {self.near_fluid_idx.shape[0]} 个入口节点")

        # 设置主管道出口条件（x ≈ length_main）- 确保它们是流体节点，但不是界面节点
        # Main Outlet (x approx length_main) - Ensure they are fluid nodes but not interface nodes
        outlet_tolerance = 1e-3
        potential_main_outlet_indices = np.nonzero(np.abs(nodes_np[:, 0] - length_main) < outlet_tolerance)[0]
        main_outlet_mask = np.isin(potential_main_outlet_indices, fluid_unique_nodes_np)
        main_outlet_candidates = potential_main_outlet_indices[main_outlet_mask]
        # 排除界面节点
        non_interface_mask = np.array([idx not in interface_nodes_set for idx in main_outlet_candidates])
        main_outlet_idx = main_outlet_candidates[non_interface_mask]
        print(f"[info] 识别主管道出口节点: 总共 {main_outlet_candidates.shape[0]} 个候选节点, 排除 {main_outlet_candidates.shape[0] - len(non_interface_mask[non_interface_mask])} 个界面节点")

        # 设置分支出口条件（在分支末端平面）- 确保它们是流体节点，但不是界面节点
        P_junction = np.array([x_junction, 0.0, 0.0])
        angle_rad = np.deg2rad(180 - angle_deg)
        V_branch_axis = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        
        P_branch_end = P_junction + length_branch * V_branch_axis
        # 距离分支末端平面近的节点: abs(dot(P - P_branch_end, V_branch_axis)) < tol
        dist_to_branch_end_plane = np.abs(np.einsum('nd,d->n', nodes_np - P_branch_end, V_branch_axis))
        
        # 节点也需要在分支管道半径内: 到轴线的距离 <= r_inner + tol
        # 计算到分支轴线的垂直距离
        Vec_P_all = nodes_np - P_junction
        proj_dist_branch = np.einsum('nd,d->n', Vec_P_all, V_branch_axis)
        Vec_proj = proj_dist_branch.reshape(-1, 1) * V_branch_axis.reshape(1, -1)
        perp_dist_branch = np.linalg.norm(Vec_P_all - Vec_proj, axis=1)
        
        branch_outlet_mask = (dist_to_branch_end_plane < outlet_tolerance) & \
                             (perp_dist_branch <= r_inner + outlet_tolerance) & \
                             (proj_dist_branch > 0)  # 确保在分支一侧

        potential_branch_outlet_indices = np.nonzero(branch_outlet_mask)[0]
        branch_outlet_fluid_mask = np.isin(potential_branch_outlet_indices, fluid_unique_nodes_np)
        branch_outlet_candidates = potential_branch_outlet_indices[branch_outlet_fluid_mask]
        # 排除界面节点
        non_interface_mask = np.array([idx not in interface_nodes_set for idx in branch_outlet_candidates])
        branch_outlet_idx = branch_outlet_candidates[non_interface_mask]
        print(f"[info] 识别分支出口节点: 总共 {branch_outlet_candidates.shape[0]} 个候选节点, 排除 {branch_outlet_candidates.shape[0] - len(non_interface_mask[non_interface_mask])} 个界面节点")
        
        # 合并主管道和分支管道出口
        combined_outlet_idx = np.concatenate((main_outlet_idx, branch_outlet_idx))
        self.outlet_fluid_idx = jnp.array(np.unique(combined_outlet_idx), dtype=jnp.int32)
        print(f"[info] 最终出口节点总数: {self.outlet_fluid_idx.shape[0]}")

        # 定义麦克风节点（在 x=1.0, y=0, z=0 附近的最近流体节点）
        # Define Mic node (closest fluid node near x=1.0, y=0, z=0 - may need adjustment)
        mic_target_pos = np.array([1.0, 0.0, 0.0])
        # 找远端节点（例如，x > 0.8 * length_main）并且是流体节点
        potential_far_indices = np.nonzero(nodes_np[:, 0] > 0.8 * length_main)[0]
        far_mask = np.isin(potential_far_indices, fluid_unique_nodes_np)
        far_fluid_idx_np = potential_far_indices[far_mask]
        self.far_fluid_idx = jnp.array(far_fluid_idx_np, dtype=jnp.int32)
        
        # 在远端流体节点中找最近的
        if len(far_fluid_idx_np) > 0:
             far_nodes_coords = nodes_np[far_fluid_idx_np]
             dists_to_mic = np.linalg.norm(far_nodes_coords - mic_target_pos, axis=1)
             self.mic_node_idx = jnp.array(far_fluid_idx_np[np.argmin(dists_to_mic)], dtype=jnp.int32)
             print(f"  麦克风节点索引: {self.mic_node_idx}, 坐标: {self.nodes[self.mic_node_idx]}")
        else:
             print("[warning] 未找到适合放置麦克风的远端流体节点.")
             self.mic_node_idx = None # 后续处理

        # --- 识别固定边界条件的固体节点 ---
        solid_node_ids_all = np.unique(solid_elements_np.flatten())
        solid_coords_all = nodes_np[solid_node_ids_all]
        solid_r_yz = np.linalg.norm(solid_coords_all[:, 1:3], axis=1)
        
        outer_radius_tol = 1e-3
        end_plane_tol = 1e-3
        
        # 查找靠近 x=0 和靠近 r=r_outer 的固体节点
        fixed_solid_mask = (np.abs(solid_coords_all[:, 0]) < end_plane_tol) & \
                           (np.abs(solid_r_yz - r_outer) < outer_radius_tol)
        
        fixed_solid_candidates = solid_node_ids_all[fixed_solid_mask]
        # 排除界面节点
        non_interface_mask = np.array([idx not in interface_nodes_set for idx in fixed_solid_candidates])
        self.fixed_solid_nodes_idx = jnp.array(fixed_solid_candidates[non_interface_mask], dtype=jnp.int32)
        print(f"[info] 识别固定固体节点: 总共 {fixed_solid_candidates.shape[0]} 个候选节点, 排除 {fixed_solid_candidates.shape[0] - len(non_interface_mask[non_interface_mask])} 个界面节点, 最终 {self.fixed_solid_nodes_idx.shape[0]} 个固定节点")
        
        if self.fixed_solid_nodes_idx.shape[0] < 3: # 通常需要至少3个非共线点
             print("[warning] 固定的固体节点少于3个。刚体模式可能未完全约束.")

        print(f"[info] 初始化完成, fluid elems: {self.fluid_elements.shape[0]}, solid elems: {self.solid_elements.shape[0]}, interface nodes: {self.interface_idx.shape[0]}, inlet nodes: {self.near_fluid_idx.shape[0]}, combined outlet nodes: {self.outlet_fluid_idx.shape[0]}, fixed solid nodes: {self.fixed_solid_nodes_idx.shape[0]}")

    def visualize_elements(self):
        """Visualize the mesh elements, boundary conditions, and interfaces using PyVista."""
        # 将节点转换为 CPU 的 numpy 数组
        nodes_np = np.array(self.nodes)  # [n_nodes, 3]

        # -----------------------------
        # 可视化 Fluid Domain (Tetrahedra)
        # -----------------------------
        fluid_cells_np = np.array(self.fluid_elements) # Shape (n_fluid_elements, 4)
        n_fluid_cells = fluid_cells_np.shape[0]
        # VTK cell format requires prepending cell size (4 for tetra)
        padding = np.full((n_fluid_cells, 1), 4, dtype=np.int64)
        vtk_fluid_cells = np.hstack((padding, fluid_cells_np)).flatten()
        # Cell types (all tetrahedra)
        vtk_fluid_cell_types = np.full(n_fluid_cells, pv.CellType.TETRA, dtype=np.int32)

        fluid_grid = pv.UnstructuredGrid(vtk_fluid_cells, vtk_fluid_cell_types, nodes_np)

        # Extract points used by fluid elements for visualization
        fluid_node_indices = np.unique(fluid_cells_np.flatten()) # Get unique node indices used in fluid elements
        fluid_points_np = nodes_np[fluid_node_indices]
        fluid_nodes_vis = pv.PolyData(fluid_points_np)

        # -----------------------------
        # 可视化 Solid Elements（四面体单元的三角面）
        # -----------------------------
        solid_cells = np.array(self.solid_elements)  # 每个单元4个节点
        solid_faces = []
        for cell in solid_cells:
            pts = cell.tolist()
            faces = [
                pts[0:3],
                [pts[0], pts[1], pts[3]],
                [pts[0], pts[2], pts[3]],
                [pts[1], pts[2], pts[3]]
            ]
            solid_faces.extend(faces)
        solid_faces_flat = []
        for face in solid_faces:
            solid_faces_flat.append(3)
            solid_faces_flat.extend(face)
        solid_faces_flat = np.array(solid_faces_flat, dtype=np.int64)

        solid_mesh = pv.PolyData(nodes_np, solid_faces_flat)

        # -----------------------------
        # 可视化 Interface
        # -----------------------------
        interface_nodes = nodes_np[np.array(self.interface_idx)]
        interface_normals = np.array(self.interface_normals)  # [n_iface, 3]

        # 构造 PyVista 点云
        interface_points = pv.PolyData(interface_nodes)

        # -----------------------------
        # 可视化 Inlet Nodes
        # -----------------------------
        inlet_nodes = nodes_np[np.array(self.near_fluid_idx)]
        inlet_points = pv.PolyData(inlet_nodes)

        # -----------------------------
        # 可视化 Outlet Nodes
        # -----------------------------
        outlet_nodes = nodes_np[np.array(self.outlet_fluid_idx)]
        outlet_points = pv.PolyData(outlet_nodes)
        
        # -----------------------------
        # 可视化 Fixed Solid Nodes
        # -----------------------------
        fixed_solid_nodes = nodes_np[np.array(self.fixed_solid_nodes_idx)]
        fixed_solid_points = pv.PolyData(fixed_solid_nodes)

        # 生成箭头：利用每个 interface 点及其法向量，设定箭头长度
        arrows = []
        arrow_length = 0.01  # 可根据需要调整
        for pt, n in zip(interface_nodes, interface_normals):
            arrow = pv.Arrow(start=pt, direction=n, scale=arrow_length)
            arrows.append(arrow)
        arrows = arrows[0].merge(arrows[1:]) if len(arrows) > 1 else arrows[0]

        # -----------------------------
        # 绘图
        # -----------------------------
        plotter = pv.Plotter()
        # Add the UnstructuredGrid for the fluid domain
        plotter.add_mesh(fluid_grid, color="cyan", opacity=0.8, show_edges=True, label="Fluid Domain (Tetrahedra)")
        # Add the nodes belonging to the fluid domain
        plotter.add_mesh(fluid_nodes_vis, color="darkblue", point_size=3, render_points_as_spheres=True, label="Fluid Nodes")
        plotter.add_mesh(solid_mesh, color="grey", opacity=0.8, label="Solid Elements")
        plotter.add_mesh(interface_points, color="blue", point_size=10, render_points_as_spheres=True, label="Interface Nodes")
        plotter.add_mesh(inlet_points, color="yellow", point_size=10, render_points_as_spheres=True, label="Inlet Nodes")
        plotter.add_mesh(outlet_points, color="magenta", point_size=10, render_points_as_spheres=True, label="Outlet Nodes")
        plotter.add_mesh(fixed_solid_points, color="purple", point_size=10, render_points_as_spheres=True, label="Fixed Solid Nodes")
        plotter.add_mesh(arrows, color="green", label="Interface Normals")
        plotter.add_legend()
        plotter.show()

    def assemble_global_fluid_system(self, source_value=None):
        """ Assemble raw fluid system A_f, F_f (Helmholtz) using local fluid mapping 
        
        Args:
            source_value: 体激励项，可以是常数或者函数f(x,y,z)
        """
        # Use pre-calculated sizes and mapping
        n_fluid_local_dof = self.N_fluid_unique # Size based on unique fluid nodes
        
        # In JAX, we'll need to be more careful about updating matrices since arrays are immutable
        # Instead of in-place updates, we'll use a different approach with more functional style
        
        # Initialize matrices with zeros
        K_f = jnp.zeros((n_fluid_local_dof, n_fluid_local_dof), dtype=jnp.float32)
        M_f = jnp.zeros((n_fluid_local_dof, n_fluid_local_dof), dtype=jnp.float32)
        F_f = jnp.zeros(n_fluid_local_dof, dtype=jnp.float32)

        print("[debug] 正在组装流体K_f和M_f(原始的，映射后)...")
        
        # Since we can't update matrices in-place in JAX, we'll first collect all contributions
        # from each element, then combine them at the end
        
        # Map global element indices to local fluid indices (precompute for all elements)
        fluid_mapping = self.fluid_mapping
        fluid_elements_np = np.array(self.fluid_elements)
        nodes_np = np.array(self.nodes)
        
        # We'll collect triplets (row, col, value) for each contribution
        K_triplets = []
        M_triplets = []
        F_triplets = []
        
        for elem_idx, elem in enumerate(tqdm(fluid_elements_np, desc="组装流体K/M", leave=False)):
            coords = nodes_np[elem]
            # Calculate element matrices
            K_e, M_e = element_matrices_fluid(jnp.array(coords))
            K_e_np = np.array(K_e)
            M_e_np = np.array(M_e)
            
            # Map global element indices to local fluid indices
            local_indices = [fluid_mapping[int(glob_idx)] for glob_idx in elem]
            
            # Collect triplets for sparse matrix assembly
            for i in range(4):
                for j in range(4):
                    row_idx = local_indices[i]
                    col_idx = local_indices[j]
                    K_triplets.append((row_idx, col_idx, K_e_np[i, j]))
                    M_triplets.append((row_idx, col_idx, M_e_np[i, j]))
            
            # 如果提供了体激励项，计算体积并更新F_f
            if source_value is not None:
                # 计算四面体体积
                v1 = coords[1] - coords[0]
                v2 = coords[2] - coords[0]
                v3 = coords[3] - coords[0]
                tetra_vol = np.abs(np.dot(np.cross(v1, v2), v3)) / 6.0
                
                # 计算四面体中心点
                centroid = np.mean(coords, axis=0)
                
                # 确定体激励项的值
                if callable(source_value):
                    # 如果体激励项是函数，在中心点求值
                    src_val = source_value(centroid[0], centroid[1], centroid[2])
                else:
                    # 否则使用常数值
                    src_val = source_value
                
                # 计算并应用到载荷向量 (collect as triplets)
                for i in range(4):
                    row_idx = local_indices[i]
                    F_triplets.append((row_idx, 0, tetra_vol * src_val / 4.0))  # 平均分配到四个节点
        
        # Now assemble the sparse matrices from triplets
        # For demonstration, we'll convert to dense matrices here
        # In a production environment, you might want to use a sparse matrix library
        
        # Sort triplets by row, then col for more efficient iteration
        K_triplets.sort(key=lambda x: (x[0], x[1]))
        M_triplets.sort(key=lambda x: (x[0], x[1]))
        F_triplets.sort(key=lambda x: x[0])
        
        # Create dense matrices from triplets (additive assembly)
        # In JAX, we'll use a vectorized approach for better performance
        # Extract rows, columns, and values from triplets
        if K_triplets:
            rows_K = jnp.array([t[0] for t in K_triplets])
            cols_K = jnp.array([t[1] for t in K_triplets])
            vals_K = jnp.array([t[2] for t in K_triplets])
            
            # Convert 2D indices to flat indices for scatter operation
            K_indices = rows_K * n_fluid_local_dof + cols_K
            K_flat = jnp.zeros(n_fluid_local_dof * n_fluid_local_dof, dtype=jnp.float32)
            K_flat = K_flat.at[K_indices].add(vals_K)
            K_f = K_flat.reshape(n_fluid_local_dof, n_fluid_local_dof)
        
        if M_triplets:
            rows_M = jnp.array([t[0] for t in M_triplets])
            cols_M = jnp.array([t[1] for t in M_triplets])
            vals_M = jnp.array([t[2] for t in M_triplets])
            
            # Convert 2D indices to flat indices for scatter operation
            M_indices = rows_M * n_fluid_local_dof + cols_M
            M_flat = jnp.zeros(n_fluid_local_dof * n_fluid_local_dof, dtype=jnp.float32)
            M_flat = M_flat.at[M_indices].add(vals_M)
            M_f = M_flat.reshape(n_fluid_local_dof, n_fluid_local_dof)
        
        if F_triplets:
            rows_F = jnp.array([t[0] for t in F_triplets])
            vals_F = jnp.array([t[2] for t in F_triplets])
            
            # Use segment_sum for efficient vector assembly
            F_f = jax.ops.segment_sum(vals_F, rows_F, n_fluid_local_dof)

        # Compute the final fluid system matrix
        k_sq = (self.omega / self.c_f)**2
        A_f = K_f - k_sq * M_f
        
        print("[debug] 原始流体系统组装完成.")
        return A_f, F_f

    def assemble_global_solid_system(self, E, nu, rho_s):
        """ Assemble raw solid system A_s, F_s using local solid mapping """
        # Use pre-calculated sizes and mapping
        n_solid_dof = self.n_solid_dof
        
        # Initialize matrices with zeros
        K_s = jnp.zeros((n_solid_dof, n_solid_dof), dtype=jnp.float32)
        M_s = jnp.zeros((n_solid_dof, n_solid_dof), dtype=jnp.float32)
        F_s = jnp.zeros(n_solid_dof, dtype=jnp.float32)

        print("[debug] 正在组装固体K_s和M_s(原始的，映射后)...")
        
        # Since we can't update matrices in-place in JAX, we'll collect triplets first 
        # then assemble the matrices at the end
        solid_mapping = self.solid_mapping
        solid_elements_np = np.array(self.solid_elements)
        nodes_np = np.array(self.nodes)
        
        # We'll collect triplets (row, col, value) for each contribution
        K_triplets = []
        M_triplets = []
        
        for elem_idx, elem in enumerate(tqdm(solid_elements_np, desc="组装固体K/M", leave=False)):
            coords = nodes_np[elem]
            
            # Convert to JAX array for the element matrix calculation
            K_e, M_e = element_matrices_solid(jnp.array(coords), E, nu, rho_s)
            K_e_np = np.array(K_e)
            M_e_np = np.array(M_e)
            
            # Map global element indices to local solid indices
            local_solid_indices = [solid_mapping[int(glob_idx)] for glob_idx in elem if int(glob_idx) in solid_mapping]
            
            # Skip this element if not all nodes are in solid_mapping
            if len(local_solid_indices) != 4:
                print(f"[Warning] 固体单元{elem}中有节点不在solid_mapping中?")
                continue
                
            # For each pair of nodes, add the corresponding 3x3 block
            for r_local_map in range(4):
                solid_idx_r = local_solid_indices[r_local_map]
                r_start = solid_idx_r * 3
                r_end = (solid_idx_r + 1) * 3
                
                for c_local_map in range(4):
                    solid_idx_c = local_solid_indices[c_local_map]
                    c_start = solid_idx_c * 3
                    c_end = (solid_idx_c + 1) * 3
                    
                    # Get the 3x3 block from K_e and M_e
                    K_block = K_e_np[r_local_map*3 : (r_local_map+1)*3, c_local_map*3 : (c_local_map+1)*3]
                    M_block = M_e_np[r_local_map*3 : (r_local_map+1)*3, c_local_map*3 : (c_local_map+1)*3]
                    
                    # Add each element of the 3x3 block as a triplet
                    for i in range(3):
                        for j in range(3):
                            row_idx = r_start + i
                            col_idx = c_start + j
                            K_triplets.append((row_idx, col_idx, K_block[i, j]))
                            M_triplets.append((row_idx, col_idx, M_block[i, j]))
        
        # Sort triplets by row, then col for more efficient iteration
        K_triplets.sort(key=lambda x: (x[0], x[1]))
        M_triplets.sort(key=lambda x: (x[0], x[1]))
        
        # Create dense matrices from triplets (additive assembly)
        # In JAX, we'll use a vectorized approach for better performance
        # Extract rows, columns, and values from triplets
        if K_triplets:
            rows_K = jnp.array([t[0] for t in K_triplets])
            cols_K = jnp.array([t[1] for t in K_triplets])
            vals_K = jnp.array([t[2] for t in K_triplets])
            
            # Convert 2D indices to flat indices for scatter operation
            K_indices = rows_K * n_solid_dof + cols_K
            K_flat = jnp.zeros(n_solid_dof * n_solid_dof, dtype=jnp.float32)
            K_flat = K_flat.at[K_indices].add(vals_K)
            K_s = K_flat.reshape(n_solid_dof, n_solid_dof)
        
        if M_triplets:
            rows_M = jnp.array([t[0] for t in M_triplets])
            cols_M = jnp.array([t[1] for t in M_triplets])
            vals_M = jnp.array([t[2] for t in M_triplets])
            
            # Convert 2D indices to flat indices for scatter operation
            M_indices = rows_M * n_solid_dof + cols_M
            M_flat = jnp.zeros(n_solid_dof * n_solid_dof, dtype=jnp.float32)
            M_flat = M_flat.at[M_indices].add(vals_M)
            M_s = M_flat.reshape(n_solid_dof, n_solid_dof)

        # Calculate A_s = K_s - omega^2 * M_s
        A_s = K_s - (self.omega**2) * M_s
        
        print("[debug] 原始固体系统组装完成.")
        return A_s, F_s

    def assemble_global_system(self, E, nu, rho_s, inlet_source=1.0, volume_source=None):
        """ Assemble coupled system using mappings and apply ALL BCs 
        
        Args:
            E, nu, rho_s: 固体材料参数
            source_value: 入口边界处声压值
            volume_source: 体激励项，可以是常数或函数f(x,y,z)
        """
        # Get sizes and mappings from self
        N_fluid_unique = self.N_fluid_unique
        n_solid_dof = self.n_solid_dof
        fluid_mapping = self.fluid_mapping
        solid_mapping = self.solid_mapping

        # ---- Raw System Assembly ----
        print("[info] 正在组装原始流体系统(映射后)...")
        A_f, F_f = self.assemble_global_fluid_system(volume_source) # Gets raw mapped A_f, F_f
        
        print("[info] 正在组装原始固体系统(映射后)...")
        A_s, F_s = self.assemble_global_solid_system(E, nu, rho_s) # Gets raw mapped A_s, F_s
        
        visualize_system(A_f, F_f, N_fluid_unique, n_solid_dof, title_suffix="Raw Fluid System Mapped")
        visualize_system(A_s, F_s, n_solid_dof, N_fluid_unique, title_suffix="Raw Solid System Mapped")
        
        # ---- Assemble Coupling Matrices (Using Mappings) ----
        print("[info] 正在组装耦合矩阵(映射后)...")
        C_sf = jnp.zeros((n_solid_dof, N_fluid_unique), dtype=jnp.float32)
        C_fs = jnp.zeros((N_fluid_unique, n_solid_dof), dtype=jnp.float32)

        if self.interface_idx.shape[0] > 0:
            # Since we can't update matrices in-place in JAX, we need to collect all contributions
            # and then apply them at once
            interface_idx_np = np.array(self.interface_idx)
            interface_nodes_set = set(interface_idx_np)
            
            # Keep track of interface normals corresponding to self.interface_idx
            interface_normals_np = np.array(self.interface_normals)
            interface_normals_map = {int(idx): normal_vec for idx, normal_vec in zip(interface_idx_np, interface_normals_np)}

            # Collect triplets for coupling matrices
            C_sf_triplets = []
            C_fs_triplets = []
            
            # Process each fluid element
            fluid_elements_np = np.array(self.fluid_elements)
            nodes_np = np.array(self.nodes)
            
            for elem_nodes in tqdm(fluid_elements_np, desc="Assembling coupling", leave=False):
                # Check if *any* node of the element is an interface node first (optimization)
                if not any(int(node) in interface_nodes_set for node in elem_nodes):
                    continue

                local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
                nodes_coords_elem = nodes_np[elem_nodes]
                
                for i_face, local_face in enumerate(local_faces):
                    global_node_indices_face = elem_nodes[np.array(local_face)]
                    
                    # Ensure all nodes of the face are in BOTH fluid and solid mappings AND on interface
                    is_mappable_interface_face = True
                    local_fluid_indices_face = []
                    local_solid_indices_face = []
                    
                    for node_idx in global_node_indices_face:
                        node_idx_int = int(node_idx)
                        if (node_idx_int in interface_nodes_set and 
                            node_idx_int in fluid_mapping and 
                            node_idx_int in solid_mapping):
                            local_fluid_indices_face.append(fluid_mapping[node_idx_int])
                            local_solid_indices_face.append(solid_mapping[node_idx_int])
                        else:
                            is_mappable_interface_face = False
                            break # Stop checking this face
                    
                    if is_mappable_interface_face:
                        # --- Calculate normal and area (same as before) ---
                        p0_idx, p1_idx, p2_idx = global_node_indices_face
                        p0, p1, p2 = nodes_np[p0_idx], nodes_np[p1_idx], nodes_np[p2_idx]
                        v1, v2 = p1 - p0, p2 - p0
                        normal_vec_cross = np.cross(v1.astype(np.float32), v2.astype(np.float32))
                        face_area = np.linalg.norm(normal_vec_cross) / 2.0

                        if face_area > 1e-12:
                            normal_vec = normal_vec_cross / (2.0 * face_area)
                            local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                            p3 = nodes_coords_elem[local_idx_p3]
                            face_centroid = (p0 + p1 + p2) / 3.0
                            vec_to_p3 = p3 - face_centroid
                            
                            if np.dot(normal_vec, vec_to_p3.astype(np.float32)) > 0:
                                normal_vec = -normal_vec
                            # --- End Normal Calculation ---

                            # Assemble C_sf (Force ON solid is -p*n)
                            force_contrib = -(face_area / 3.0) * normal_vec
                            
                            # Assemble C_fs (Fluid equation term rho*omega^2*u_n)
                            motion_contrib = rho_f * (self.omega**2) * (face_area / 3.0) * normal_vec

                            # Add contributions using LOCAL indices
                            for i_node_face in range(3): # Iterate 0, 1, 2 for the face nodes
                                fluid_local_idx = local_fluid_indices_face[i_node_face]
                                solid_local_idx = local_solid_indices_face[i_node_face]
                                
                                # Add to C_sf for each DOF (3 per solid node)
                                for i_dof in range(3):
                                    C_sf_triplets.append((solid_local_idx*3 + i_dof, fluid_local_idx, force_contrib[i_dof]))
                                
                                # Add to C_fs for each DOF
                                for i_dof in range(3):
                                    C_fs_triplets.append((fluid_local_idx, solid_local_idx*3 + i_dof, motion_contrib[i_dof]))
            
            # Now assemble coupling matrices from triplets
            # Sort triplets for more efficient iteration
            C_sf_triplets.sort(key=lambda x: (x[0], x[1]))
            C_fs_triplets.sort(key=lambda x: (x[0], x[1]))
            
            # Create matrices from triplets (additive assembly)
            if C_sf_triplets:
                rows_sf = jnp.array([t[0] for t in C_sf_triplets])
                cols_sf = jnp.array([t[1] for t in C_sf_triplets])
                vals_sf = jnp.array([t[2] for t in C_sf_triplets])
                
                # Convert 2D indices to flat indices for scatter operation
                sf_indices = rows_sf * N_fluid_unique + cols_sf
                sf_flat = jnp.zeros(n_solid_dof * N_fluid_unique, dtype=jnp.float32)
                sf_flat = sf_flat.at[sf_indices].add(vals_sf)
                C_sf = sf_flat.reshape(n_solid_dof, N_fluid_unique)
            
            if C_fs_triplets:
                rows_fs = jnp.array([t[0] for t in C_fs_triplets])
                cols_fs = jnp.array([t[1] for t in C_fs_triplets])
                vals_fs = jnp.array([t[2] for t in C_fs_triplets])
                
                # Convert 2D indices to flat indices for scatter operation
                fs_indices = rows_fs * n_solid_dof + cols_fs
                fs_flat = jnp.zeros(N_fluid_unique * n_solid_dof, dtype=jnp.float32)
                fs_flat = fs_flat.at[fs_indices].add(vals_fs)
                C_fs = fs_flat.reshape(N_fluid_unique, n_solid_dof)
        else:
            print("[info] 未找到界面节点，跳过耦合矩阵组装.")

        # Check for zero coupling
        if jnp.all(C_fs == 0) and jnp.all(C_sf == 0):
            print("[warning] 耦合矩阵全为零！")
            
        # ---- Construct Global Block Matrix (Mapped) ----
        print("[info] 正在构造全局块矩阵(映射后)...")
        global_dim = N_fluid_unique + n_solid_dof

        # Check dimensions before creating A_global
        if A_f.shape[0] != N_fluid_unique or A_f.shape[1] != N_fluid_unique:
            print(f"[Error] A_f形状不匹配！预期({N_fluid_unique},{N_fluid_unique})，实际得到{A_f.shape}")
            raise ValueError("A_f维度不匹配")
            
        if A_s.shape[0] != n_solid_dof or A_s.shape[1] != n_solid_dof:
            print(f"[Error] A_s形状不匹配！预期({n_solid_dof},{n_solid_dof})，实际得到{A_s.shape}")
            raise ValueError("A_s维度不匹配")
            
        if C_fs.shape[0] != N_fluid_unique or C_fs.shape[1] != n_solid_dof:
            print(f"[Error] C_fs形状不匹配！预期({N_fluid_unique},{n_solid_dof})，实际得到{C_fs.shape}")
            raise ValueError("C_fs维度不匹配")
            
        if C_sf.shape[0] != n_solid_dof or C_sf.shape[1] != N_fluid_unique:
            print(f"[Error] C_sf形状不匹配！预期({n_solid_dof},{N_fluid_unique})，实际得到{C_sf.shape}")
            raise ValueError("C_sf维度不匹配")

        # Construct global matrix in JAX by assembling blocks
        # Create empty matrix
        A_global = jnp.zeros((global_dim, global_dim), dtype=jnp.float32)
        
        # Fill in the blocks
        A_global = A_global.at[:N_fluid_unique, :N_fluid_unique].set(A_f)
        A_global = A_global.at[:N_fluid_unique, N_fluid_unique:].set(C_fs)
        A_global = A_global.at[N_fluid_unique:, :N_fluid_unique].set(C_sf)
        A_global = A_global.at[N_fluid_unique:, N_fluid_unique:].set(A_s)

        # ---- Construct Global Force Vector (Mapped) ----
        # Ensure F_f and F_s have the correct shape
        if F_f.shape[0] != N_fluid_unique:
            print(f"[Warning] F_f形状不匹配！预期({N_fluid_unique},)，实际得到{F_f.shape}")
            F_f = jnp.resize(F_f, (N_fluid_unique,))
            
        if F_s.shape[0] != n_solid_dof:
            print(f"[Warning] F_s形状不匹配！预期({n_solid_dof},)，实际得到{F_s.shape}")
            F_s = jnp.resize(F_s, (n_solid_dof,))
        
        # Concatenate vectors
        F_global = jnp.concatenate([F_f, F_s])
        
        print(f"[debug] 原始映射后的A_global形状: {A_global.shape}, 原始映射后的F_global形状: {F_global.shape}")

        # <-- Call visualization BEFORE applying BCs -->
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="Raw Assembly Mapped")

        # ---- Apply ALL Dirichlet Boundary Conditions to Mapped A_global, F_global ----
        print("[info] 正在应用边界条件(映射后)...")
        penalty = self.bcpenalty

        # Apply Fluid Inlet BC (p = source_value) - Dirichlet condition
        print(f"[debug] 正在应用流体入口边界条件(p={inlet_source})到{self.near_fluid_idx.shape[0]}个全局节点...")
        
        # In JAX, we need to handle boundary conditions differently due to immutability
        # We'll use triplets to collect all boundary modifications, then apply them at once
        bc_modifications = []
        
        # Track nodes processed by inlet BC
        nodes_processed_by_inlet = set()
        near_fluid_idx_np = np.array(self.near_fluid_idx)
        
        for global_idx in near_fluid_idx_np:
            global_idx_int = int(global_idx)
            if global_idx_int in fluid_mapping:
                local_idx = fluid_mapping[global_idx_int]
                if local_idx not in nodes_processed_by_inlet:
                    # Clear row and column
                    bc_modifications.append(('clear_row', local_idx))
                    bc_modifications.append(('clear_col', local_idx))
                    
                    # Set diagonal and right-hand side
                    bc_modifications.append(('set_diag', local_idx, penalty))
                    bc_modifications.append(('set_rhs', local_idx, penalty * inlet_source))
                    
                    nodes_processed_by_inlet.add(local_idx)
            else:
                print(f"[warning] 入口节点{global_idx_int}在fluid_mapping中未找到.")

        # Apply Fluid Outlet BC (dp/dn = 0) - Neumann condition (non-reflecting)
        # For Neumann boundary condition, we don't need to modify the system matrix
        # The natural boundary condition dp/dn = 0 is automatically satisfied
        print(f"[info] 在出口处使用诺依曼边界条件(dp/dn=0)({self.outlet_fluid_idx.shape[0]}个节点)")
        print(f"[info] 这是一个无反射出口边界条件")
        
        # Verify that outlet nodes are in the fluid domain
        valid_outlet_count = 0
        outlet_fluid_idx_np = np.array(self.outlet_fluid_idx)
        
        for global_idx in outlet_fluid_idx_np:
            global_idx_int = int(global_idx)
            if global_idx_int in fluid_mapping:
                valid_outlet_count += 1
            else:
                print(f"[warning] 出口节点{global_idx_int}在fluid_mapping中未找到.")
                
        print(f"[info] 找到{valid_outlet_count}个用于诺依曼边界条件的有效出口节点")

        # Apply Solid Fixed BCs (u = 0)
        print(f"[debug] 正在应用固定固体边界条件到{self.fixed_solid_nodes_idx.shape[0]}个全局节点...")
        processed_solid_global_dofs = set()
        fixed_solid_nodes_idx_np = np.array(self.fixed_solid_nodes_idx)
        
        for global_node_idx in fixed_solid_nodes_idx_np:
            global_node_idx_int = int(global_node_idx)
            if global_node_idx_int in solid_mapping:
                solid_local_idx = solid_mapping[global_node_idx_int]
                # Calculate global DOF indices with N_fluid_unique offset
                global_dof_indices = [N_fluid_unique + solid_local_idx*3 + i for i in range(3)]

                for dof_idx in global_dof_indices:
                    if dof_idx < global_dim:
                        if dof_idx not in processed_solid_global_dofs:
                            bc_modifications.append(('clear_row', dof_idx))
                            bc_modifications.append(('clear_col', dof_idx))
                            bc_modifications.append(('set_diag', dof_idx, penalty))
                            bc_modifications.append(('set_rhs', dof_idx, 0.0))
                            processed_solid_global_dofs.add(dof_idx)
                    else:
                        print(f"[warning] 计算得到的固体自由度索引{dof_idx}超出边界({global_dim}).")
            else:
                print(f"[warning] 固定的固体节点{global_node_idx_int}在solid_mapping中未找到.")

        # Now apply all boundary condition modifications
        # In JAX, we need to create a new matrix and vector rather than modifying in-place
        A_with_bc = A_global
        F_with_bc = F_global
        
        # Group modifications by type for vectorized operations
        clear_row_indices = [item[1] for item in bc_modifications if item[0] == 'clear_row']
        clear_col_indices = [item[1] for item in bc_modifications if item[0] == 'clear_col']
        diag_updates = [(item[1], item[2]) for item in bc_modifications if item[0] == 'set_diag']
        rhs_updates = [(item[1], item[2]) for item in bc_modifications if item[0] == 'set_rhs']
        
        # Apply row clearing in one operation if there are any
        if clear_row_indices:
            # Create a mask for all rows that need to be cleared
            row_indices = jnp.array(clear_row_indices)
            # Use advanced indexing to clear multiple rows at once
            A_with_bc = A_with_bc.at[row_indices, :].set(0.0)
        
        # Apply column clearing in one operation if there are any
        if clear_col_indices:
            # Create a mask for all columns that need to be cleared
            col_indices = jnp.array(clear_col_indices)
            # Use advanced indexing to clear multiple columns at once
            A_with_bc = A_with_bc.at[:, col_indices].set(0.0)
        
        # Apply diagonal updates in one operation if there are any
        if diag_updates:
            diag_indices = jnp.array([idx for idx, _ in diag_updates])
            diag_values = jnp.array([val for _, val in diag_updates])
            # Update diagonal entries
            A_with_bc = A_with_bc.at[diag_indices, diag_indices].set(diag_values)
        
        # Apply right-hand side updates in one operation if there are any
        if rhs_updates:
            rhs_indices = jnp.array([idx for idx, _ in rhs_updates])
            rhs_values = jnp.array([val for _, val in rhs_updates])
            # Update RHS entries
            F_with_bc = F_with_bc.at[rhs_indices].set(rhs_values)

        print("[info] 全局矩阵与边界条件组装完成.")
        # <-- Optionally visualize AFTER applying BCs -->
        visualize_system(A_with_bc, F_with_bc, N_fluid_unique, n_solid_dof, title_suffix="With BCs Mapped")

        return A_with_bc, F_with_bc, N_fluid_unique, n_solid_dof

    def solve(self, E, nu, rho_s, volume_source=None):
        """
        给定材料参数，组装全局系统并求解，返回：
          - 预测远端麦克风处流体声压（取 fluid 域 x≈1.0 点平均）
          - 全局解向量 u（其中 u[0:n_nodes] 为 fluid 声压）
          
        Args:
            E, nu, rho_s: 固体材料参数
            volume_source: 体激励项，可以是常数或函数f(x,y,z)
                           例如：constant_source = 10.0
                           或者：def spatial_source(x, y, z): return 10.0 * np.exp(-(x**2 + y**2 + z**2))
                           
        Examples:
            # 使用常数体激励项
            solver.solve(E=1e9, nu=0.3, rho_s=1000.0, volume_source=10.0)
            
            # 使用空间变化的体激励项
            def gaussian_source(x, y, z):
                # 高斯分布的声源，在(0.5, 0, 0)位置最强
                return 10.0 * jnp.exp(-5*((x-0.5)**2 + y**2 + z**2))
            
            solver.solve(E=1e9, nu=0.3, rho_s=1000.0, volume_source=gaussian_source)
        """
        A_global, F_global, N_fluid_unique, n_solid_dof_actual = self.assemble_global_system(
            E, nu, rho_s, inlet_source=0.0, volume_source=volume_source)
        
        print("[info] 开始求解 (mapped system)")
        
        # JAX's linear solver can be used for this
        # Here we use linalg.solve which is suitable for dense matrices
        # For very large systems, you might want to use a more specialized solver
        try:
            # JAX's solve might be more efficient and differentiation-friendly than lstsq
            u = jax.scipy.linalg.solve(A_global, F_global)
            
            # Alternative: use least squares for potentially better conditioning
            # u = jax.scipy.linalg.lstsq(A_global, F_global)[0]
        except Exception as e:
            print(f"求解器错误: {e}")
            print("矩阵可能仍然是奇异的或条件数不佳.")
            # Save matrices for debugging
            np.save('A_global_error.npy', np.array(A_global))
            np.save('F_global_error.npy', np.array(F_global))
            raise
            
        print("[info] 求解完成")
        
        # Extract microphone pressure using fluid_mapping
        p_mic = jnp.array(0.0, dtype=u.dtype)  # Default value
        
        if self.mic_node_idx is not None:
            global_mic_idx = int(self.mic_node_idx)
            if global_mic_idx in self.fluid_mapping:
                local_mic_idx = self.fluid_mapping[global_mic_idx]
                if local_mic_idx < N_fluid_unique:  # Double check bounds
                    p_mic = u[local_mic_idx]
                else:
                    print(f"[warning] 映射后的麦克风索引{local_mic_idx}超出流体自由度范围({N_fluid_unique}).")
            else:
                print(f"[warning] 全局麦克风节点索引{global_mic_idx}在fluid_mapping中未找到.")
        else:
            print("[warning] 麦克风节点索引未定义.")
            
        print(f"[info] 预测远端麦克风处流体声压: {p_mic.squeeze()}")
        return p_mic.squeeze(), u  # Return scalar p_mic
