# jax_fem.py
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
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

# 定义流体密度
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
    # 使用叉积和点积计算混合积
    vol = jnp.abs(jnp.dot(jnp.cross(v0, v1), v2)) / 6.0
    return vol

@jit
def compute_shape_function_gradients(coords):
    """
    计算四面体线性形函数梯度（常数）。
    方法：对齐次线性系统 [1 x y z] 的逆矩阵的后三行。
    返回: [4,3]，每一行为对应节点形函数在 x,y,z 方向的梯度。
    """
    ones = jnp.ones((4,1))
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
    K_e = V * (grads @ grads.transpose())
    ones_4 = jnp.ones((4,4))
    M_e = V / 20.0 * (jnp.eye(4) + ones_4)
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
    ])
    return D

@jit
def compute_B_matrix(coords):
    """
    构造四面体单元的应变-位移矩阵 B (6 x 12)。
    利用线性形函数梯度计算，各节点 3 dof。
    """
    grads = compute_shape_function_gradients(coords)  # [4,3]
    B = jnp.zeros((6, 12))
    
    # 对每个节点分别构造B矩阵块
    for i in range(4):
        bx, by, bz = grads[i]
        B_i = jnp.array([
            [bx,   0,    0],
            [0,   by,    0],
            [0,    0,   bz],
            [by,  bx,    0],
            [0,   bz,   by],
            [bz,   0,   bx]
        ])
        B = B.at[:, i*3:(i+1)*3].set(B_i)
    
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
    K_e = V * (B.transpose() @ (D @ B))
    # 质量矩阵采用对角 lumped mass:
    M_e = jnp.zeros((12, 12))
    m_lump = rho_s * V / 4.0
    
    # 构造对角块质量矩阵
    for i in range(4):
        start_idx = i*3
        end_idx = (i+1)*3
        M_e_block = m_lump * jnp.eye(3)
        M_e = M_e.at[start_idx:end_idx, start_idx:end_idx].set(M_e_block)
    
    return K_e, M_e

def visualize_system(A, f, n_nodes, n_solid_dof, title_suffix="Raw System"):
    """
    Visualizes the matrix A using colored scatter plot (log scale) and vector f as a heatmap.
    Marks the divisions between fluid and solid DOFs.
    """
    print(f"[Visualizing System] 矩阵形状: {A.shape}, 向量形状: {f.shape}")
    print(f"[Visualizing System] 流体自由度(n_nodes): {n_nodes}, 固体自由度: {n_solid_dof}")
    total_dof = n_nodes + n_solid_dof
    
    # 将JAX数组转换为NumPy供可视化使用
    if hasattr(A, 'device_buffer'):
        A_np = np.array(A)
        f_np = np.array(f)
    else:
        A_np = A
        f_np = f
    
    if A_np.shape[0] != total_dof or A_np.shape[1] != total_dof or f_np.shape[0] != total_dof:
        print(f"[Warning] visualize_system中的维度不匹配: A={A_np.shape}, f={f_np.shape}, 预期总自由度={total_dof}")
        actual_total_dof = A_np.shape[0]
        if n_nodes > actual_total_dof: n_nodes = actual_total_dof
    else:
        actual_total_dof = total_dof

    # Convert dense array to scipy sparse COO matrix for easier access to values/coords
    A_sparse_coo = sps.coo_matrix(A_np)
    
    # Filter out small values based on tolerance
    tol = 1e-9
    mask = np.abs(A_sparse_coo.data) > tol
    row = A_sparse_coo.row[mask]
    col = A_sparse_coo.col[mask]
    data = A_sparse_coo.data[mask]
    
    num_non_zero = len(data)
    print(f"[Visualizing System] 矩阵稀疏度: {num_non_zero / (A_np.shape[0] * A_np.shape[1]) * 100:.4f}% 非零元素 > {tol}.")
    
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
    ax.set_title(f"Matrix A ({A_np.shape[0]}x{A_np.shape[1]}) Non-Zero Elements (Log Color Scale)")
    ax.set_xlabel("Column Index")
    ax.set_ylabel("Row Index")
    ax.set_xlim(-0.5, A_np.shape[1]-0.5) # Adjust xlim slightly for text
    ax.set_ylim(A_np.shape[0]-0.5, -0.5) # Adjust ylim slightly for text
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
    # Reshape f_np to (N, 1) for matshow
    f_plot = f_np.reshape(-1, 1)
    img = ax.matshow(f_plot, cmap='coolwarm', aspect='auto') # Use 'auto' aspect for vector
    ax.set_title(f"Vector f ({f_np.shape[0]}x1)")
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
        self.nodes = jnp.array(mesh.points[:, :3])
        
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
        # 在JAX中要使用numpy处理唯一值和索引创建
        fluid_elements_flat = self.fluid_elements.flatten()
        solid_elements_flat = self.solid_elements.flatten()
        
        fluid_unique_nodes, fluid_inverse_indices = np.unique(np.array(fluid_elements_flat), return_inverse=True)
        solid_unique_nodes, solid_inverse_indices = np.unique(np.array(solid_elements_flat), return_inverse=True)
        
        self.fluid_unique_nodes = jnp.array(fluid_unique_nodes)
        self.solid_unique_nodes = jnp.array(solid_unique_nodes)

        self.N_fluid_unique = len(self.fluid_unique_nodes)
        self.N_solid_unique = len(self.solid_unique_nodes)
        self.n_solid_dof = self.N_solid_unique * 3

        # 全局索引 -> 局部索引映射字典
        self.fluid_mapping = {int(global_idx): local_idx for local_idx, global_idx in enumerate(fluid_unique_nodes)}
        self.solid_mapping = {int(global_idx): local_idx for local_idx, global_idx in enumerate(solid_unique_nodes)}

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
            
            # 转换为array
            self.interface_idx = jnp.array(list(interface_nodes_set), dtype=jnp.int32)
            print(f"[info] 从界面物理组找到 {len(interface_nodes_set)} 个流固界面节点")
        else:
            # 如果找不到界面物理组，退回到找流体和固体共享的节点
            print("[info] 未找到界面物理组，退回到通过流体和固体共享节点识别界面...")
            fluid_node_set = set(np.array(self.fluid_unique_nodes))
            solid_node_set = set(np.array(self.solid_unique_nodes))
            interface_node_set = fluid_node_set.intersection(solid_node_set)
            self.interface_idx = jnp.array(list(interface_node_set), dtype=jnp.int32)
            print(f"[info] 找到 {len(interface_node_set)} 个流固界面共享节点")

        # --- 计算界面法向量 ---
        print("[info] 计算界面法向量...")
        # 创建界面节点到法向量的映射（使用Python字典）
        node_to_face_normals = {int(node_id): [] for node_id in self.interface_idx}
        
        # 由于JAX的函数式特性，这部分计算保留使用循环
        fluid_elements_np = np.array(self.fluid_elements)
        nodes_np = np.array(self.nodes)
        interface_idx_np = np.array(self.interface_idx)
        
        # 迭代流体单元找到界面面及其法向量
        for elem_nodes in tqdm(fluid_elements_np, desc="计算界面法向量"):
            nodes_coords = nodes_np[elem_nodes]  # 4个节点坐标
            # 定义四个面（局部索引）
            local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]

            for i_face, local_face in enumerate(local_faces):
                global_node_indices = np.array([elem_nodes[i] for i in local_face])  # 面的全局节点索引
                
                # 检查是否所有3个节点都是界面节点
                is_interface_face = all(node in node_to_face_normals for node in global_node_indices)

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
            
        self.interface_normals = jnp.array(final_normals_list)
        
        # 过滤掉法向量为零的节点
        # 由于JAX数组不可变，这里先用NumPy处理
        normal_magnitudes = np.linalg.norm(np.array(self.interface_normals), axis=1)
        norm_tolerance = 1e-9
        valid_normal_mask = normal_magnitudes > norm_tolerance
        
        original_count = len(self.interface_idx)
        self.interface_idx = jnp.array(np.array(self.interface_idx)[valid_normal_mask])
        self.interface_normals = jnp.array(np.array(self.interface_normals)[valid_normal_mask])
        
        filtered_count = len(self.interface_idx)
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"[info] 移除了 {removed_count} 个法向量无效的界面节点 (norm < {norm_tolerance}).")
        print(f"[info] 最终界面节点数量: {filtered_count}")
        
        # 创建界面节点集合，用于排除它们不被用作其他边界条件
        interface_nodes_set = set(np.array(self.interface_idx))

        # --- Inlet/Outlet Definitions ---
        # 为了保持与原版一致的功能，边界处理部分保留NumPy实现
        nodes_np = np.array(self.nodes)
        fluid_unique_nodes_np = np.array(self.fluid_unique_nodes)
        
        # 设置入口条件（x ≈ 0）- 确保它们是流体节点，但不是界面节点
        potential_near_indices = np.nonzero(np.abs(nodes_np[:, 0]) < 1e-3)[0]
        near_mask = np.isin(potential_near_indices, fluid_unique_nodes_np)
        near_fluid_candidates = potential_near_indices[near_mask]
        # 排除界面节点
        non_interface_mask = np.array([idx not in interface_nodes_set for idx in near_fluid_candidates])
        self.near_fluid_idx = jnp.array(near_fluid_candidates[non_interface_mask])
        print(f"[info] 识别入口节点: 总共 {len(near_fluid_candidates)} 个候选节点, 排除 {len(near_fluid_candidates) - len(self.near_fluid_idx)} 个界面节点, 最终 {len(self.near_fluid_idx)} 个入口节点")

        # 设置主管道出口条件（x ≈ length_main）- 确保它们是流体节点，但不是界面节点
        outlet_tolerance = 1e-3
        potential_main_outlet_indices = np.nonzero(np.abs(nodes_np[:, 0] - length_main) < outlet_tolerance)[0]
        main_outlet_mask = np.isin(potential_main_outlet_indices, fluid_unique_nodes_np)
        main_outlet_candidates = potential_main_outlet_indices[main_outlet_mask]
        # 排除界面节点
        non_interface_mask = np.array([idx not in interface_nodes_set for idx in main_outlet_candidates])
        main_outlet_idx = main_outlet_candidates[non_interface_mask]
        print(f"[info] 识别主管道出口节点: 总共 {len(main_outlet_candidates)} 个候选节点, 排除 {len(main_outlet_candidates) - len(main_outlet_idx)} 个界面节点")

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
        Vec_proj = proj_dist_branch[:, np.newaxis] * V_branch_axis[np.newaxis, :]
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
        print(f"[info] 识别分支出口节点: 总共 {len(branch_outlet_candidates)} 个候选节点, 排除 {len(branch_outlet_candidates) - len(branch_outlet_idx)} 个界面节点")
        
        # 合并主管道和分支管道出口
        combined_outlet_idx = np.concatenate((main_outlet_idx, branch_outlet_idx))
        self.outlet_fluid_idx = jnp.array(np.unique(combined_outlet_idx))
        print(f"[info] 最终出口节点总数: {len(self.outlet_fluid_idx)}")
        
        # 定义麦克风节点（在 x=1.0, y=0, z=0 附近的最近流体节点）
        mic_target_pos = np.array([1.0, 0.0, 0.0])
        # 找远端节点（例如，x > 0.8 * length_main）并且是流体节点
        potential_far_indices = np.nonzero(nodes_np[:, 0] > 0.8 * length_main)[0]
        far_mask = np.isin(potential_far_indices, fluid_unique_nodes_np)
        self.far_fluid_idx = jnp.array(potential_far_indices[far_mask])
        
        # 在远端流体节点中找最近的
        if len(self.far_fluid_idx) > 0:
             far_fluid_idx_np = np.array(self.far_fluid_idx)
             far_nodes_coords = nodes_np[far_fluid_idx_np]
             dists_to_mic = np.linalg.norm(far_nodes_coords - mic_target_pos, axis=1)
             self.mic_node_idx = self.far_fluid_idx[np.argmin(dists_to_mic)] # 单个节点索引
             print(f"  麦克风节点索引: {self.mic_node_idx}, 坐标: {self.nodes[int(self.mic_node_idx)]}")
        else:
             print("[warning] 未找到适合放置麦克风的远端流体节点.")
             self.mic_node_idx = None # 后续处理

        # --- 识别固定边界条件的固体节点 ---
        solid_node_ids_all = np.unique(np.array(self.solid_elements).flatten())
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
        self.fixed_solid_nodes_idx = jnp.array(fixed_solid_candidates[non_interface_mask])
        print(f"[info] 识别固定固体节点: 总共 {len(fixed_solid_candidates)} 个候选节点, 排除 {len(fixed_solid_candidates) - len(self.fixed_solid_nodes_idx)} 个界面节点, 最终 {len(self.fixed_solid_nodes_idx)} 个固定节点")
        
        if len(self.fixed_solid_nodes_idx) < 3: # 通常需要至少3个非共线点
             print("[warning] 固定的固体节点少于3个。刚体模式可能未完全约束.")

        print(f"[info] 初始化完成, fluid elems: {self.fluid_elements.shape[0]}, solid elems: {self.solid_elements.shape[0]}, interface nodes: {len(self.interface_idx)}, inlet nodes: {len(self.near_fluid_idx)}, combined outlet nodes: {len(self.outlet_fluid_idx)}, fixed solid nodes: {len(self.fixed_solid_nodes_idx)}")
        # self.visualize_elements()
        # input("[info] 按回车继续...") 

    def visualize_elements(self):
        # 将JAX节点转换为NumPy数组
        nodes_np = np.array(self.nodes)  # [n_nodes, 3]
        fluid_elements_np = np.array(self.fluid_elements)
        solid_elements_np = np.array(self.solid_elements)
        interface_idx_np = np.array(self.interface_idx)
        interface_normals_np = np.array(self.interface_normals)
        near_fluid_idx_np = np.array(self.near_fluid_idx)
        outlet_fluid_idx_np = np.array(self.outlet_fluid_idx)
        fixed_solid_nodes_idx_np = np.array(self.fixed_solid_nodes_idx)

        # -----------------------------
        # 可视化 Fluid Domain (Tetrahedra)
        # -----------------------------
        n_fluid_cells = fluid_elements_np.shape[0]
        # VTK cell format requires prepending cell size (4 for tetra)
        padding = np.full((n_fluid_cells, 1), 4, dtype=np.int64)
        vtk_fluid_cells = np.hstack((padding, fluid_elements_np)).flatten()
        # Cell types (all tetrahedra)
        vtk_fluid_cell_types = np.full(n_fluid_cells, pv.CellType.TETRA, dtype=np.int32)

        fluid_grid = pv.UnstructuredGrid(vtk_fluid_cells, vtk_fluid_cell_types, nodes_np)

        # Extract points used by fluid elements for visualization
        fluid_node_indices = np.unique(fluid_elements_np.flatten()) # Get unique node indices used in fluid elements
        fluid_points_np = nodes_np[fluid_node_indices]
        fluid_nodes_vis = pv.PolyData(fluid_points_np)

        # -----------------------------
        # 可视化 Solid Elements（四面体单元的三角面）
        # -----------------------------
        solid_faces = []
        for cell in solid_elements_np:
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
        interface_nodes = nodes_np[interface_idx_np]
        interface_normals = interface_normals_np  # [n_iface, 3]

        # 构造 PyVista 点云
        interface_points = pv.PolyData(interface_nodes)

        # -----------------------------
        # 可视化 Inlet Nodes
        # -----------------------------
        inlet_nodes = nodes_np[near_fluid_idx_np]
        inlet_points = pv.PolyData(inlet_nodes)

        # -----------------------------
        # 可视化 Outlet Nodes
        # -----------------------------
        outlet_nodes = nodes_np[outlet_fluid_idx_np]
        outlet_points = pv.PolyData(outlet_nodes)
        
        # -----------------------------
        # 可视化 Fixed Solid Nodes
        # -----------------------------
        fixed_solid_nodes = nodes_np[fixed_solid_nodes_idx_np]
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
    
    def assemble_global_system(self, E, nu, rho_s, inlet_source=1.0, volume_source=None):
        """ 
        组装耦合系统，应用所有边界条件
        
        Args:
            E, nu, rho_s: 固体材料参数
            inlet_source: 入口边界处声压值
            volume_source: 体激励项，可以是常数或函数f(x,y,z)
        """
        # 获取各个尺寸和映射
        N_fluid_unique = self.N_fluid_unique
        n_solid_dof = self.n_solid_dof
        fluid_mapping = self.fluid_mapping
        solid_mapping = self.solid_mapping

        # ---- 原始系统组装 ----
        print("[info] 正在组装原始流体系统(映射后)...")
        A_f, F_f = self.assemble_global_fluid_system(volume_source) # 获取原始映射的 A_f, F_f
        print("[info] 正在组装原始固体系统(映射后)...")
        A_s, F_s = self.assemble_global_solid_system(E, nu, rho_s) # 获取原始映射的 A_s, F_s
        
        visualize_system(A_f, F_f, N_fluid_unique, n_solid_dof, title_suffix="Raw Fluid System Mapped")
        visualize_system(A_s, F_s, n_solid_dof, N_fluid_unique, title_suffix="Raw Solid System Mapped")
        
        # ---- 组装耦合矩阵 ----
        print("[info] 正在组装耦合矩阵(映射后)...")
        # 初始化耦合矩阵
        C_sf = jnp.zeros((n_solid_dof, N_fluid_unique))
        C_fs = jnp.zeros((N_fluid_unique, n_solid_dof))

        # 由于JAX的特性，这段组装矩阵的代码难以直接转换为函数式风格
        # 我们使用NumPy辅助处理
        C_sf_np = np.zeros((n_solid_dof, N_fluid_unique))
        C_fs_np = np.zeros((N_fluid_unique, n_solid_dof))
        
        fluid_elements_np = np.array(self.fluid_elements)
        nodes_np = np.array(self.nodes)
        interface_idx_np = np.array(self.interface_idx)
        interface_normals_np = np.array(self.interface_normals)
        
        if len(interface_idx_np) > 0:
            interface_node_set = set(interface_idx_np)
            # 创建界面法向量映射字典
            interface_normals_map = {int(idx): normal for idx, normal in zip(interface_idx_np, interface_normals_np)}

            for elem_nodes in tqdm(fluid_elements_np, desc="组装耦合矩阵"):
                # 检查元素是否有界面节点（优化）
                if not any(node in interface_node_set for node in elem_nodes):
                    continue

                local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
                nodes_coords_elem = nodes_np[elem_nodes]
                for i_face, local_face in enumerate(local_faces):
                    global_node_indices_face = elem_nodes[local_face]
                    # 确保面的所有节点都在流体和固体映射中且是界面节点
                    is_mappable_interface_face = True
                    local_fluid_indices_face = []
                    local_solid_indices_face = []

                    for node_idx in global_node_indices_face:
                        if (node_idx in interface_node_set and 
                            node_idx in fluid_mapping and 
                            node_idx in solid_mapping):
                            local_fluid_indices_face.append(fluid_mapping[node_idx])
                            local_solid_indices_face.append(solid_mapping[node_idx])
                        else:
                            is_mappable_interface_face = False
                            break  # 停止检查这个面

                    if is_mappable_interface_face:
                        # --- 计算法向量和面积 ---
                        p0_idx, p1_idx, p2_idx = global_node_indices_face
                        p0, p1, p2 = nodes_np[p0_idx], nodes_np[p1_idx], nodes_np[p2_idx]
                        v1, v2 = p1 - p0, p2 - p0
                        normal_vec_cross = np.cross(v1, v2)
                        face_area = np.linalg.norm(normal_vec_cross) / 2.0

                        if face_area > 1e-12:
                            normal_vec = normal_vec_cross / (2.0 * face_area)
                            local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                            p3 = nodes_coords_elem[local_idx_p3]
                            face_centroid = (p0 + p1 + p2) / 3.0
                            vec_to_p3 = p3 - face_centroid
                            if np.dot(normal_vec, vec_to_p3) > 0: 
                                normal_vec = -normal_vec
                            # --- 法向量计算结束 ---

                            # 组装 C_sf (Force ON solid is -p*n)
                            force_contrib = -(face_area / 3.0) * normal_vec
                            # 组装 C_fs (Fluid equation term rho*omega^2*u_n)
                            motion_contrib = rho_f * (self.omega**2) * (face_area / 3.0) * normal_vec

                            # 使用局部索引添加贡献
                            for i_node_face in range(3): # 遍历面上的三个节点
                                fluid_local_idx = local_fluid_indices_face[i_node_face]
                                solid_local_idx = local_solid_indices_face[i_node_face]

                                C_sf_np[solid_local_idx*3:(solid_local_idx+1)*3, fluid_local_idx] += force_contrib
                                C_fs_np[fluid_local_idx, solid_local_idx*3:(solid_local_idx+1)*3] += motion_contrib
        else:
            print("[info] 未找到界面节点，跳过耦合矩阵组装.")

        # 转换回JAX数组
        C_sf = jnp.array(C_sf_np)
        C_fs = jnp.array(C_fs_np)

        # 检查耦合矩阵是否全零
        if jnp.all(C_fs == 0) and jnp.all(C_sf == 0):
            print("[warning] 耦合矩阵全为零！")
            
        # ---- 构造全局矩阵 ----
        print("[info] 正在构造全局块矩阵(映射后)...")
        global_dim = N_fluid_unique + n_solid_dof

        # 检查维度匹配
        if A_f.shape != (N_fluid_unique, N_fluid_unique):
            print(f"[Error] A_f形状不匹配！预期({N_fluid_unique},{N_fluid_unique})，实际得到{A_f.shape}")
            raise ValueError("A_f维度不匹配")
        if A_s.shape != (n_solid_dof, n_solid_dof):
            print(f"[Error] A_s形状不匹配！预期({n_solid_dof},{n_solid_dof})，实际得到{A_s.shape}")
            raise ValueError("A_s维度不匹配")
        if C_fs.shape != (N_fluid_unique, n_solid_dof):
            print(f"[Error] C_fs形状不匹配！预期({N_fluid_unique},{n_solid_dof})，实际得到{C_fs.shape}")
            raise ValueError("C_fs维度不匹配")
        if C_sf.shape != (n_solid_dof, N_fluid_unique):
            print(f"[Error] C_sf形状不匹配！预期({n_solid_dof},{N_fluid_unique})，实际得到{C_sf.shape}")
            raise ValueError("C_sf维度不匹配")

        # 构造全局系统矩阵（JAX中，我们可以使用block_matrix函数）
        A_global = jnp.zeros((global_dim, global_dim))
        # 填充四个块
        A_global = A_global.at[:N_fluid_unique, :N_fluid_unique].set(A_f)
        A_global = A_global.at[:N_fluid_unique, N_fluid_unique:].set(C_fs)
        A_global = A_global.at[N_fluid_unique:, :N_fluid_unique].set(C_sf)
        A_global = A_global.at[N_fluid_unique:, N_fluid_unique:].set(A_s)

        # ---- 构造全局载荷向量 ----
        # 确保F_f和F_s维度正确
        if F_f.shape[0] != N_fluid_unique:
            F_f = F_f.reshape(N_fluid_unique)
        if F_s.shape[0] != n_solid_dof:
            F_s = F_s.reshape(n_solid_dof)
        
        F_global = jnp.concatenate((F_f, F_s))
        print(f"[debug] 原始映射后的A_global形状: {A_global.shape}, 原始映射后的F_global形状: {F_global.shape}")

        # 在应用边界条件前可视化
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="Raw Assembly Mapped")

        # ---- 应用所有边界条件 ----
        print("[info] 正在应用边界条件(映射后)...")
        penalty = self.bcpenalty
        
        # 由于JAX的不可变性质，我们需要使用NumPy处理边界条件应用
        A_global_np = np.array(A_global)
        F_global_np = np.array(F_global)
        
        # 应用流体入口边界条件 (p = source_value)
        print(f"[debug] 正在应用流体入口边界条件(p={inlet_source})到{len(self.near_fluid_idx)}个全局节点...")
        nodes_processed_by_inlet = set()
        
        for global_idx in np.array(self.near_fluid_idx):
            global_idx_int = int(global_idx)
            if global_idx_int in fluid_mapping:  # 确保是流体节点
                local_idx = fluid_mapping[global_idx_int]  # 获取局部流体索引
                if local_idx not in nodes_processed_by_inlet:
                    A_global_np[local_idx, :] = 0.0
                    A_global_np[:, local_idx] = 0.0
                    A_global_np[local_idx, local_idx] = penalty
                    F_global_np[local_idx] = penalty * inlet_source
                    nodes_processed_by_inlet.add(local_idx)
            else:
                print(f"[warning] 入口节点{global_idx_int}在fluid_mapping中未找到.")

        # 应用流体出口边界条件 (dp/dn = 0) - 诺依曼边界条件
        print(f"[info] 在出口处使用诺依曼边界条件(dp/dn=0)({len(self.outlet_fluid_idx)}个节点)")
        print(f"[info] 这是一个无反射出口边界条件")
        
        # 验证出口节点在流体域中
        valid_outlet_count = 0
        for global_idx in np.array(self.outlet_fluid_idx):
            global_idx_int = int(global_idx)
            if global_idx_int in fluid_mapping:
                valid_outlet_count += 1
            else:
                print(f"[warning] 出口节点{global_idx_int}在fluid_mapping中未找到.")
        print(f"[info] 找到{valid_outlet_count}个用于诺依曼边界条件的有效出口节点")

        # 应用固体固定边界条件 (u = 0)
        print(f"[debug] 正在应用固定固体边界条件到{len(self.fixed_solid_nodes_idx)}个全局节点...")
        processed_solid_global_dofs = set()
        
        for global_node_idx in np.array(self.fixed_solid_nodes_idx):
            global_node_idx_int = int(global_node_idx)
            if global_node_idx_int in solid_mapping:
                solid_local_idx = solid_mapping[global_node_idx_int]
                # 计算全局自由度索引（考虑N_fluid_unique偏移）
                global_dof_indices = [N_fluid_unique + solid_local_idx*3 + i for i in range(3)]

                for dof_idx in global_dof_indices:
                    if dof_idx < global_dim:
                        if dof_idx not in processed_solid_global_dofs and A_global_np[dof_idx, dof_idx] != penalty:
                            A_global_np[dof_idx, :] = 0.0
                            A_global_np[:, dof_idx] = 0.0
                            A_global_np[dof_idx, dof_idx] = penalty
                            F_global_np[dof_idx] = 0.0
                            processed_solid_global_dofs.add(dof_idx)
                    else:
                        print(f"[warning] 计算得到的固体自由度索引{dof_idx}超出边界({global_dim}).")
            else:
                print(f"[warning] 固定的固体节点{global_node_idx_int}在solid_mapping中未找到.")

        # 将修改后的矩阵和向量转回JAX数组
        A_global = jnp.array(A_global_np)
        F_global = jnp.array(F_global_np)

        print("[info] 全局矩阵与边界条件组装完成.")
        # 应用边界条件后可视化
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="With BCs Mapped")

        return A_global, F_global, N_fluid_unique, n_solid_dof

    def solve(self, E, nu, rho_s, volume_source=None):
        """
        给定材料参数，组装全局系统并求解，返回：
          - 预测远端麦克风处流体声压（取 fluid 域 x≈1.0 点平均）
          - 全局解向量 u（其中 u[0:n_nodes] 为 fluid 声压）
          
        Args:
            E, nu, rho_s: 固体材料参数
            volume_source: 体激励项，可以是常数或函数f(x,y,z)
        """
        A_global, F_global, N_fluid_unique, n_solid_dof_actual = self.assemble_global_system(
            E, nu, rho_s, inlet_source=0.0, volume_source=volume_source
        )
        
        print("[info] 开始求解 (mapped system)")
        try:
            # 使用JAX的线性代数求解器
            u = jnp.linalg.solve(A_global, F_global)
        except Exception as e:
            print(f"求解器错误: {e}")
            print("矩阵可能仍然是奇异的或条件数不佳.")
            # 保存矩阵为NumPy格式
            np.save('A_global_error.npy', np.array(A_global))
            np.save('F_global_error.npy', np.array(F_global))
            raise
            
        print("[info] 求解完成")
        # 提取前k个最大值进行打印
        topk_indices = jnp.argsort(-jnp.abs(u))[:100]
        topk_values = u[topk_indices]
        print(f"[info] 解向量前100大值: {topk_values}")
        
        # 提取麦克风处的压力
        p_mic = jnp.array(0.0)  # 默认值
        
        if self.mic_node_idx is not None:
            global_mic_idx = int(self.mic_node_idx)
            if global_mic_idx in self.fluid_mapping:
                local_mic_idx = self.fluid_mapping[global_mic_idx]
                if local_mic_idx < N_fluid_unique:  # 检查索引边界
                    p_mic = u[local_mic_idx]
                else:
                    print(f"[warning] 映射后的麦克风索引{local_mic_idx}超出流体自由度范围({N_fluid_unique}).")
            else:
                print(f"[warning] 全局麦克风节点索引{global_mic_idx}在fluid_mapping中未找到.")
        else:
            print("[warning] 麦克风节点索引未定义.")
            
        print(f"[info] 预测远端麦克风处流体声压: {p_mic}")
        return p_mic, u  # 返回麦克风压力标量和全局解向量

    def assemble_global_fluid_system(self, source_value=None):
        """ 
        组装原始流体系统 A_f, F_f (Helmholtz方程)，使用局部流体映射
        
        Args:
            source_value: 体激励项，可以是常数或者函数f(x,y,z)
        """
        # 使用预计算的尺寸和映射
        n_fluid_local_dof = self.N_fluid_unique  # 基于唯一流体节点的尺寸
        fluid_mapping = self.fluid_mapping

        # 由于JAX的函数式风格，矩阵组装使用NumPy辅助完成
        K_f_np = np.zeros((n_fluid_local_dof, n_fluid_local_dof))
        M_f_np = np.zeros((n_fluid_local_dof, n_fluid_local_dof))
        F_f_np = np.zeros(n_fluid_local_dof)

        # 获取NumPy版本的节点和元素
        nodes_np = np.array(self.nodes)
        fluid_elements_np = np.array(self.fluid_elements)

        print("[debug] 正在组装流体K_f和M_f(原始的，映射后)...")
        for elem in tqdm(fluid_elements_np, desc="组装流体K/M"):
            coords = nodes_np[elem]
            # 使用JAX计算局部矩阵
            K_e, M_e = element_matrices_fluid(jnp.array(coords))
            # 转换为NumPy用于组装
            K_e_np = np.array(K_e)
            M_e_np = np.array(M_e)
            
            # 将全局元素索引映射到局部流体索引
            local_indices = [fluid_mapping[int(glob_idx)] for glob_idx in elem]

            # 使用局部索引分散组装
            for r_local_map in range(4):  # 局部索引0-3
                row_idx = local_indices[r_local_map]
                for c_local_map in range(4):
                    col_idx = local_indices[c_local_map]
                    K_f_np[row_idx, col_idx] += K_e_np[r_local_map, c_local_map]
                    M_f_np[row_idx, col_idx] += M_e_np[r_local_map, c_local_map]
            
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
                
                # 计算并应用到载荷向量
                for r_local_map in range(4):
                    row_idx = local_indices[r_local_map]
                    F_f_np[row_idx] += tetra_vol * src_val / 4.0  # 平均分配到四个节点

        # 计算A_f
        k_sq = (self.omega / self.c_f)**2
        
        # 转换回JAX数组
        K_f = jnp.array(K_f_np)
        M_f = jnp.array(M_f_np)
        F_f = jnp.array(F_f_np)
        
        A_f = K_f - k_sq * M_f
        print("[debug] 原始流体系统组装完成.")
        return A_f, F_f

    def assemble_global_solid_system(self, E, nu, rho_s):
        """ 组装原始固体系统 A_s, F_s，使用局部固体映射 """
        # 使用预计算的尺寸和映射
        n_solid_dof = self.n_solid_dof
        solid_mapping = self.solid_mapping

        # 使用NumPy辅助组装
        K_s_np = np.zeros((n_solid_dof, n_solid_dof))
        M_s_np = np.zeros((n_solid_dof, n_solid_dof))
        F_s_np = np.zeros(n_solid_dof)

        # 获取NumPy版本的节点和元素
        nodes_np = np.array(self.nodes)
        solid_elements_np = np.array(self.solid_elements)

        print("[debug] 正在组装固体K_s和M_s(原始的，映射后)...")
        for elem in tqdm(solid_elements_np, desc="组装固体K/M"):
            coords = nodes_np[elem]
            # 使用JAX计算局部矩阵
            K_e, M_e = element_matrices_solid(jnp.array(coords), E, nu, rho_s)
            # 转换为NumPy用于组装
            K_e_np = np.array(K_e)
            M_e_np = np.array(M_e)
            
            # 将全局元素索引映射到局部固体索引
            try:
                local_solid_indices = [solid_mapping[int(glob_idx)] for glob_idx in elem]
            except KeyError:
                print(f"[Warning] 固体单元{elem}中有节点不在solid_mapping中?")
                continue
            
            # 检查所有节点是否映射成功
            if len(local_solid_indices) != 4:
                print(f"[Warning] 固体单元{elem}中有节点不在solid_mapping中?")
                continue

            for r_local_map in range(4):  # 索引0-3，表示元素节点
                solid_idx_r = local_solid_indices[r_local_map]  # 局部固体索引
                for c_local_map in range(4):
                    solid_idx_c = local_solid_indices[c_local_map]
                    # 从K_e和M_e获取3x3块，基于元素节点顺序
                    K_block = K_e_np[r_local_map*3:(r_local_map+1)*3, c_local_map*3:(c_local_map+1)*3]
                    M_block = M_e_np[r_local_map*3:(r_local_map+1)*3, c_local_map*3:(c_local_map+1)*3]
                    # 添加到全局固体矩阵，使用映射索引
                    K_s_np[solid_idx_r*3:(solid_idx_r+1)*3, solid_idx_c*3:(solid_idx_c+1)*3] += K_block
                    M_s_np[solid_idx_r*3:(solid_idx_r+1)*3, solid_idx_c*3:(solid_idx_c+1)*3] += M_block

        # 计算A_s = K_s - omega^2 * M_s
        # 转换回JAX数组
        K_s = jnp.array(K_s_np)
        M_s = jnp.array(M_s_np)
        F_s = jnp.array(F_s_np)
        
        A_s = K_s - (self.omega**2) * M_s
        print("[debug] 原始固体系统组装完成.")
        return A_s, F_s 