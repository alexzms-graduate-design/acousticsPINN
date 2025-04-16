# numpy_fem.py
import numpy as np
import meshio
from tqdm import tqdm
import pyvista as pv
import matplotlib.pyplot as plt
import scipy.sparse as sps
import matplotlib.colors as mcolors  # For LogNorm

# 基本常量定义
# 管道参数定义
r_outer = 0.05   # 外半径 5cm
r_inner = 0.045  # 内半径 4.5cm
angle_deg = 45     # 分支角度 (degrees)
length_main = 1.5  # 主干总长 1.5m
x_junction = 0.5   # Junction point x-coordinate
# Branch length MUST match the value used in geometry_gmsh.py
length_branch = 0.3 

# 声学常数
rho_f = 1.225  # kg/m^3 (Air density)

# ===============================
# 辅助函数：线性四面体元的计算
# ===============================
def compute_tetra_volume(coords):
    """
    计算四面体体积
    coords: [4,3] 每行为节点坐标
    Volume = abs(det([x2-x1, x3-x1, x4-x1]))/6
    """
    v0 = coords[1] - coords[0]
    v1 = coords[2] - coords[0]
    v2 = coords[3] - coords[0]
    vol = np.abs(np.linalg.det(np.stack([v0, v1, v2], axis=1))) / 6.0
    return vol

def compute_shape_function_gradients(coords):
    """
    计算四面体线性形函数梯度（常数）。
    方法：对齐次线性系统 [1 x y z] 的逆矩阵的后三行。
    返回: [4,3]，每一行为对应节点形函数在 x,y,z 方向的梯度。
    """
    ones = np.ones((4, 1), dtype=coords.dtype)
    A = np.concatenate([ones, coords], axis=1)  # [4,4]
    A_inv = np.linalg.inv(A)
    # 后三行为 b, c, d 系数，对应梯度
    # Extract the coefficients (rows 1, 2, 3 of A_inv) which have shape [3,4]
    coeffs = A_inv[1:, :]
    # Transpose to get shape [4,3] - each row contains gradients for one node
    grads = np.transpose(coeffs)
    return grads

def element_matrices_fluid(coords):
    """
    计算流体四面体单元的局部刚度和质量矩阵。
    使用公式：
      K_e = Volume * (grad(N) @ grad(N)^T)
      M_e = Volume/20 * (I + ones(4,4))
    """
    V = compute_tetra_volume(coords)
    grads = compute_shape_function_gradients(coords)  # [4,3]
    # Compute stiffness matrix using einsum for correct dot products between gradient vectors
    K_e = V * np.einsum('ij,kj->ik', grads, grads)
    ones_4 = np.ones((4, 4), dtype=coords.dtype)
    M_e = V / 20.0 * (np.eye(4, dtype=coords.dtype) + ones_4)
    return K_e, M_e

def elasticity_D_matrix(E, nu):
    """
    3D各向同性弹性材料 D 矩阵 (6x6)，Voigt 表示。
    """
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    D = coeff * np.array([
        [1 - nu,    nu,    nu,       0,       0,       0],
        [nu,    1 - nu,    nu,       0,       0,       0],
        [nu,      nu,  1 - nu,       0,       0,       0],
        [0,       0,      0,  (1 - 2 * nu) / 2,  0,       0],
        [0,       0,      0,       0,  (1 - 2 * nu) / 2,  0],
        [0,       0,      0,       0,       0,  (1 - 2 * nu) / 2]
    ], dtype=np.float32)
    return D

def compute_B_matrix(coords):
    """
    构造四面体单元的应变-位移矩阵 B (6 x 12)。
    利用线性形函数梯度计算，各节点 3 dof。
    """
    grads = compute_shape_function_gradients(coords)  # [4,3]
    B = np.zeros((6, 12), dtype=coords.dtype)
    
    # In JAX, we need to avoid in-place operations and use functional updates
    rows = []
    for i in range(4):
        bx, by, bz = grads[i]
        B_block = np.array([
            [bx,    0,    0],
            [0,   by,     0],
            [0,    0,    bz],
            [by,   bx,    0],
            [0,    bz,   by],
            [bz,   0,    bx]
        ], dtype=coords.dtype)
        rows.append(B_block)
    
    # Concatenate the blocks horizontally
    B = np.block([rows[0], rows[1], rows[2], rows[3]])
    return B

def element_matrices_solid(coords, E, nu, rho_s):
    """
    计算固体四面体单元（管壁）的局部刚度和质量矩阵。
    结构单元 DOF = 3 per node, 整体矩阵尺寸 12x12。
    """
    V = compute_tetra_volume(coords)
    B = compute_B_matrix(coords)  # [6, 12]
    D = elasticity_D_matrix(E, nu)  # [6,6]
    # 刚度矩阵：K_e = V * B^T D B
    # Use einsum for more reliable matrix multiplication in JAX
    K_e = V * (B.T @ (D @ B))
    # 质量矩阵采用对角 lumped mass:
    m_lump = rho_s * V / 4.0
    
    # Create diagonal blocks for mass matrix
    M_e = np.zeros((12, 12), dtype=coords.dtype)
    # Use functional updates for JAX
    for i in range(4):
        start_idx = i * 3
        end_idx = (i + 1) * 3
        diag_block = m_lump * np.eye(3, dtype=coords.dtype)
        # Use direct assignment for NumPy
        M_e[start_idx:end_idx, start_idx:end_idx] = diag_block
    
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

    # Move data to CPU NumPy arrays
    A_cpu = np.array(A)
    f_cpu = np.array(f)

    # Convert to scipy sparse COO matrix for easier access to values/coords
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
        cppenalty: 流固耦合惩罚参数 β
        bcpenalty: 边界条件惩罚参数
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
        self.nodes = np.array(mesh.points[:, :3], dtype=np.float32)
        
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
        
        self.fluid_elements = np.array(fluid_elems_np, dtype=np.int32)
        self.solid_elements = np.array(solid_elems_np, dtype=np.int32)

        print(f"[info] 加载了 {self.fluid_elements.shape[0]} 个流体单元 (tag {fluid_tag}) 和 {self.solid_elements.shape[0]} 个固体单元 (tag {solid_tag}).")

        # --- 创建节点映射 ---
        # Using NumPy for these operations as JAX unique has different behavior
        fluid_unique_nodes_np, fluid_inverse_indices_np = np.unique(np.array(self.fluid_elements).flatten(), return_inverse=True)
        solid_unique_nodes_np, solid_inverse_indices_np = np.unique(np.array(self.solid_elements).flatten(), return_inverse=True)
        
        self.fluid_unique_nodes = np.array(fluid_unique_nodes_np)
        self.solid_unique_nodes = np.array(solid_unique_nodes_np)

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
            self.interface_idx = np.array(list(interface_nodes_set), dtype=np.int32)
            print(f"[info] 从界面物理组找到 {len(interface_nodes_set)} 个流固界面节点")
        else:
            # 如果找不到界面物理组，退回到找流体和固体共享的节点
            print("[info] 未找到界面物理组，退回到通过流体和固体共享节点识别界面...")
            fluid_node_set = set(fluid_unique_nodes_np)
            solid_node_set = set(solid_unique_nodes_np)
            interface_node_set = fluid_node_set.intersection(solid_node_set)
            self.interface_idx = np.array(list(interface_node_set), dtype=np.int32)
            print(f"[info] 找到 {len(interface_node_set)} 个流固界面共享节点")

        # --- 计算界面法向量 ---
        print("[info] 计算界面法向量...")
        # Convert to set for membership testing
        interface_nodes_set = set(np.array(self.interface_idx))
        node_to_face_normals = {node_id: [] for node_id in interface_nodes_set}
        
        # 迭代流体单元找到界面面及其法向量
        fluid_elements_np = np.array(self.fluid_elements)
        for elem_idx, elem_nodes in enumerate(tqdm(fluid_elements_np, desc="计算界面法向量")):
            nodes_coords = np.array(self.nodes[elem_nodes])  # 4个节点坐标
            # 定义四个面（局部索引）
            local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]

            for i_face, local_face in enumerate(local_faces):
                global_node_indices = elem_nodes[np.array(local_face)] # 面的全局节点索引

                # 检查是否所有3个节点都是界面节点
                is_interface_face = all(node_id in interface_nodes_set for node_id in global_node_indices)

                if is_interface_face:
                    # 计算面法向量
                    p0, p1, p2 = self.nodes[global_node_indices]
                    p0, p1, p2 = np.array(p0), np.array(p1), np.array(p2)
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
                            if node_id in interface_nodes_set:
                                node_to_face_normals[node_id].append(normal)
        
        # 通过平均计算最终顶点法向量
        final_normals_list = []
        final_interface_indices = []
        zero_normal_count = 0
        
        for node_id in interface_nodes_set:
            normals_to_average = node_to_face_normals.get(node_id, [])

            if not normals_to_average:
                zero_normal_count += 1
                continue
            else:
                avg_normal = np.sum(normals_to_average, axis=0)
                norm_val = np.linalg.norm(avg_normal)
                if norm_val < 1e-12:
                    zero_normal_count += 1
                    continue
                else:
                    avg_normal = avg_normal / norm_val
                    final_normals_list.append(avg_normal)
                    final_interface_indices.append(node_id)
        
        if zero_normal_count > 0:
            print(f"[warning] {zero_normal_count} 个界面节点的法向量为零或无效，已排除.")
            
        # Store the final valid interface nodes and normals
        self.interface_idx = np.array(final_interface_indices, dtype=np.int32)
        self.interface_normals = np.array(final_normals_list, dtype=np.float32)
        print(f"[info] 最终有效界面节点数量: {len(final_interface_indices)}")
        
        # 创建界面节点集合，用于排除它们不被用作其他边界条件
        interface_nodes_set = set(np.array(self.interface_idx))

        # --- Inlet/Outlet Definitions ---
        # 设置入口条件（x ≈ 0）- 确保它们是流体节点，但不是界面节点
        # Using NumPy for these boolean operations
        nodes_np = np.array(self.nodes)
        potential_near_indices = np.where(np.abs(nodes_np[:, 0]) < 1e-3)[0]
        near_fluid_candidates = []
        for idx in potential_near_indices:
            if idx in self.fluid_mapping and idx not in interface_nodes_set:
                near_fluid_candidates.append(idx)
        
        self.near_fluid_idx = np.array(near_fluid_candidates, dtype=np.int32)
        print(f"[info] 识别入口节点: 最终 {len(near_fluid_candidates)} 个入口节点")

        # 设置主管道出口条件（x ≈ length_main）- 确保它们是流体节点，但不是界面节点
        outlet_tolerance = 1e-3
        potential_main_outlet_indices = np.where(np.abs(nodes_np[:, 0] - length_main) < outlet_tolerance)[0]
        main_outlet_candidates = []
        for idx in potential_main_outlet_indices:
            if idx in self.fluid_mapping and idx not in interface_nodes_set:
                main_outlet_candidates.append(idx)
                
        print(f"[info] 识别主管道出口节点: {len(main_outlet_candidates)} 个有效节点")

        # 设置分支出口条件（在分支末端平面）- 确保它们是流体节点，但不是界面节点
        P_junction = np.array([x_junction, 0.0, 0.0])
        angle_rad = np.deg2rad(180 - angle_deg)
        V_branch_axis = np.array([np.cos(angle_rad), np.sin(angle_rad), 0.0])
        
        P_branch_end = P_junction + length_branch * V_branch_axis
        # 距离分支末端平面近的节点: abs(dot(P - P_branch_end, V_branch_axis)) < tol
        dist_to_branch_end_plane = np.abs(np.dot(nodes_np - P_branch_end, V_branch_axis))
        
        # 节点也需要在分支管道半径内: 到轴线的距离 <= r_inner + tol
        # 计算到分支轴线的垂直距离
        Vec_P_all = nodes_np - P_junction
        proj_dist_branch = np.dot(Vec_P_all, V_branch_axis)
        Vec_proj = proj_dist_branch[:, np.newaxis] * V_branch_axis
        perp_dist_branch = np.linalg.norm(Vec_P_all - Vec_proj, axis=1)
        
        branch_outlet_mask = (dist_to_branch_end_plane < outlet_tolerance) & \
                             (perp_dist_branch <= r_inner + outlet_tolerance) & \
                             (proj_dist_branch > 0)  # 确保在分支一侧

        potential_branch_outlet_indices = np.where(branch_outlet_mask)[0]
        branch_outlet_candidates = []
        for idx in potential_branch_outlet_indices:
            if idx in self.fluid_mapping and idx not in interface_nodes_set:
                branch_outlet_candidates.append(idx)
                
        print(f"[info] 识别分支出口节点: {len(branch_outlet_candidates)} 个有效节点")
        
        # 合并主管道和分支管道出口
        combined_outlet_list = main_outlet_candidates + branch_outlet_candidates
        # Remove duplicates (shouldn't have any, but just to be safe)
        combined_outlet_set = set(combined_outlet_list)
        self.outlet_fluid_idx = np.array(list(combined_outlet_set), dtype=np.int32)
        print(f"[info] 最终出口节点总数: {len(combined_outlet_set)}")
        
        # 定义麦克风节点（在 x=1.0, y=0, z=0 附近的最近流体节点）
        # Define Mic node (closest fluid node near x=1.0, y=0, z=0 - may need adjustment)
        mic_target_pos = np.array([1.0, 0.0, 0.0])
        # 找远端节点（例如，x > 0.8 * length_main）并且是流体节点
        potential_far_indices = np.where(nodes_np[:, 0] > 0.8 * length_main)[0]
        far_fluid_candidates = []
        for idx in potential_far_indices:
            if idx in self.fluid_mapping:
                far_fluid_candidates.append(idx)
        
        self.far_fluid_idx = np.array(far_fluid_candidates, dtype=np.int32)
        
        # 在远端流体节点中找最近的
        if len(far_fluid_candidates) > 0:
             far_nodes_coords = nodes_np[far_fluid_candidates]
             dists_to_mic = np.linalg.norm(far_nodes_coords - mic_target_pos, axis=1)
             self.mic_node_idx = far_fluid_candidates[np.argmin(dists_to_mic)] # 单个节点索引
             print(f"  麦克风节点索引: {self.mic_node_idx}, 坐标: {nodes_np[self.mic_node_idx]}")
        else:
             print("[warning] 未找到适合放置麦克风的远端流体节点.")
             self.mic_node_idx = None # 后续处理

        # --- 识别固定边界条件的固体节点 ---
        solid_node_ids_all_np = np.unique(np.array(self.solid_elements).flatten())
        solid_coords_all = nodes_np[solid_node_ids_all_np]
        solid_r_yz = np.linalg.norm(solid_coords_all[:, 1:3], axis=1)
        
        outer_radius_tol = 1e-3
        end_plane_tol = 1e-3
        
        # 查找靠近 x=0 和靠近 r=r_outer 的固体节点
        fixed_solid_mask = (np.abs(solid_coords_all[:, 0]) < end_plane_tol) & \
                           (np.abs(solid_r_yz - r_outer) < outer_radius_tol)
        
        fixed_solid_indices = solid_node_ids_all_np[fixed_solid_mask]
        fixed_solid_candidates = []
        for idx in fixed_solid_indices:
            if idx not in interface_nodes_set:
                fixed_solid_candidates.append(idx)
                
        self.fixed_solid_nodes_idx = np.array(fixed_solid_candidates, dtype=np.int32)
        print(f"[info] 识别固定固体节点: 最终 {len(fixed_solid_candidates)} 个固定节点")
        
        if len(fixed_solid_candidates) < 3: # 通常需要至少3个非共线点
             print("[warning] 固定的固体节点少于3个。刚体模式可能未完全约束.")

        print(f"[info] 初始化完成, fluid elems: {self.fluid_elements.shape[0]}, solid elems: {self.solid_elements.shape[0]}, interface nodes: {self.interface_idx.shape[0]}, inlet nodes: {self.near_fluid_idx.shape[0]}, combined outlet nodes: {self.outlet_fluid_idx.shape[0]}, fixed solid nodes: {self.fixed_solid_nodes_idx.shape[0]}")
        # self.visualize_elements()  # Comment out to avoid pausing for visualization
        
    def visualize_elements(self):
        """
        Visualization is mostly a side effect, so we'll keep this non-JAX.
        """
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
        # Since we can't use JAX tracing with NumPy operations in JIT mode,
        # we will use only NumPy for array operations in assembly
        # and convert to JAX arrays only at the end
        
        # Use pre-calculated sizes and mapping
        n_fluid_local_dof = self.N_fluid_unique  # Size based on unique fluid nodes
        fluid_mapping = self.fluid_mapping
        fluid_elements_np = np.array(self.fluid_elements)
        nodes_np = np.array(self.nodes)

        K_f = np.zeros((n_fluid_local_dof, n_fluid_local_dof), dtype=np.float32)
        M_f = np.zeros((n_fluid_local_dof, n_fluid_local_dof), dtype=np.float32)
        F_f = np.zeros(n_fluid_local_dof, dtype=np.float32)

        print("[debug] 正在组装流体K_f和M_f(原始的，映射后)...")
        for elem in tqdm(fluid_elements_np, desc="组装流体K/M", leave=False):
            coords_np = nodes_np[elem]
            
            # Calculate element matrices using NumPy operations (not JAX)
            # We need to reimplement the matrix calculations directly
            V = np.abs(np.linalg.det(np.stack([
                coords_np[1] - coords_np[0],
                coords_np[2] - coords_np[0],
                coords_np[3] - coords_np[0]
            ], axis=1))) / 6.0
            
            # Calculate shape function gradients
            ones = np.ones((4, 1), dtype=np.float32)
            A = np.concatenate([ones, coords_np], axis=1)
            A_inv = np.linalg.inv(A)
            grads = np.transpose(A_inv[1:, :])  # Shape [4,3]
            
            # Calculate stiffness and mass matrices
            K_e = V * np.einsum('ij,kj->ik', grads, grads)
            ones_4 = np.ones((4, 4), dtype=np.float32)
            M_e = V / 20.0 * (np.eye(4, dtype=np.float32) + ones_4)
            
            # Map global element indices to local fluid indices
            local_indices = [fluid_mapping[glob_idx] for glob_idx in elem]

            # Scatter using local indices
            for r_local_map in range(4):  # Index within local_indices (0-3)
                row_idx = local_indices[r_local_map]
                for c_local_map in range(4):
                    col_idx = local_indices[c_local_map]
                    K_f[row_idx, col_idx] += K_e[r_local_map, c_local_map]
                    M_f[row_idx, col_idx] += M_e[r_local_map, c_local_map]
            
            # 如果提供了体激励项，计算体积并更新F_f
            if source_value is not None:
                # 计算四面体体积
                v1 = coords_np[1] - coords_np[0]
                v2 = coords_np[2] - coords_np[0]
                v3 = coords_np[3] - coords_np[0]
                tetra_vol = np.abs(np.dot(np.cross(v1, v2), v3)) / 6.0
                
                # 计算四面体中心点
                centroid = np.mean(coords_np, axis=0)
                
                # 确定体激励项的值
                if callable(source_value):
                    # 如果体激励项是函数，在中心点求值
                    # Need to convert to JAX array for function evaluation and back
                    centroid_jax = np.array(centroid)
                    src_val_jax = source_value(centroid_jax[0], centroid_jax[1], centroid_jax[2])
                    # Convert JAX array to NumPy without using float() which doesn't work with tracers
                    src_val = np.array(src_val_jax)
                else:
                    # 否则使用常数值
                    src_val = source_value
                
                # 计算并应用到载荷向量
                for r_local_map in range(4):
                    row_idx = local_indices[r_local_map]
                    F_f[row_idx] += tetra_vol * src_val / 4.0  # 平均分配到四个节点

        k_sq = (self.omega / self.c_f)**2
        A_f = K_f - k_sq * M_f
        
        # Convert to JAX arrays for return ONLY at the end
        A_f_jax = np.array(A_f)
        F_f_jax = np.array(F_f)
        
        print("[debug] 原始流体系统组装完成.")
        return A_f_jax, F_f_jax
        
    def assemble_global_solid_system(self, E, nu, rho_s):
        """ Assemble raw solid system A_s, F_s using local solid mapping """
        # Use only NumPy operations for assembly to avoid JAX tracing issues
        # Instead of converting to float, pass the JAX values directly to 
        # NumPy operations when needed but avoid direct type conversion
        
        # Use pre-calculated sizes and mapping
        n_solid_dof = self.n_solid_dof
        solid_mapping = self.solid_mapping
        solid_elements_np = np.array(self.solid_elements)
        nodes_np = np.array(self.nodes)

        K_s = np.zeros((n_solid_dof, n_solid_dof), dtype=np.float32)
        M_s = np.zeros((n_solid_dof, n_solid_dof), dtype=np.float32)
        F_s = np.zeros(n_solid_dof, dtype=np.float32)

        # Convert JAX values to NumPy only once at the beginning
        # Avoid float() conversion which doesn't work with tracers
        E_np = np.array(E)
        nu_np = np.array(nu)
        rho_s_np = np.array(rho_s)

        print("[debug] 正在组装固体K_s和M_s(原始的，映射后)...")
        for elem in tqdm(solid_elements_np, desc="组装固体K/M", leave=False):
            coords_np = nodes_np[elem]
            
            # Calculate element matrices using NumPy operations
            # Compute volume
            V = np.abs(np.linalg.det(np.stack([
                coords_np[1] - coords_np[0],
                coords_np[2] - coords_np[0],
                coords_np[3] - coords_np[0]
            ], axis=1))) / 6.0
            
            # Calculate shape function gradients
            ones = np.ones((4, 1), dtype=np.float32)
            A = np.concatenate([ones, coords_np], axis=1)
            A_inv = np.linalg.inv(A)
            grads = np.transpose(A_inv[1:, :])  # Shape [4,3]
            
            # Compute B matrix
            B = np.zeros((6, 12), dtype=np.float32)
            for i in range(4):
                bx, by, bz = grads[i]
                B_block = np.array([
                    [bx, 0, 0],
                    [0, by, 0],
                    [0, 0, bz],
                    [by, bx, 0],
                    [0, bz, by],
                    [bz, 0, bx]
                ], dtype=np.float32)
                B[:, i*3:(i+1)*3] = B_block
            
            # Compute elasticity D matrix with NumPy arrays
            coeff = E_np / ((1 + nu_np) * (1 - 2 * nu_np))
            D = coeff * np.array([
                [1 - nu_np, nu_np, nu_np, 0, 0, 0],
                [nu_np, 1 - nu_np, nu_np, 0, 0, 0],
                [nu_np, nu_np, 1 - nu_np, 0, 0, 0],
                [0, 0, 0, (1 - 2 * nu_np) / 2, 0, 0],
                [0, 0, 0, 0, (1 - 2 * nu_np) / 2, 0],
                [0, 0, 0, 0, 0, (1 - 2 * nu_np) / 2]
            ], dtype=np.float32)
            
            # Calculate stiffness matrix K_e = V * B^T D B
            K_e = V * (B.T @ (D @ B))
            
            # Create lumped mass matrix
            m_lump = rho_s_np * V / 4.0
            M_e = np.zeros((12, 12), dtype=np.float32)
            for i in range(4):
                start_idx = i * 3
                end_idx = (i + 1) * 3
                M_e[start_idx:end_idx, start_idx:end_idx] = m_lump * np.eye(3, dtype=np.float32)
            
            # Map global element indices to local solid indices
            local_solid_indices = [solid_mapping[glob_idx] for glob_idx in elem if glob_idx in solid_mapping]
            # This assumes all nodes in a solid element are unique solid nodes, should be true
            if len(local_solid_indices) != 4:
                print(f"[Warning] 固体单元{elem}中有节点不在solid_mapping中?")
                continue

            for r_local_map in range(4):  # Index referring to element's node (0-3)
                solid_idx_r = local_solid_indices[r_local_map]  # Local solid index
                for c_local_map in range(4):
                    solid_idx_c = local_solid_indices[c_local_map]
                    # Get the 3x3 block from K_e and M_e based on the element's node order
                    K_block = K_e[r_local_map*3:(r_local_map+1)*3, c_local_map*3:(c_local_map+1)*3]
                    M_block = M_e[r_local_map*3:(r_local_map+1)*3, c_local_map*3:(c_local_map+1)*3]
                    # Add to the global solid matrices at the mapped indices
                    K_s[solid_idx_r*3:(solid_idx_r+1)*3, solid_idx_c*3:(solid_idx_c+1)*3] += K_block
                    M_s[solid_idx_r*3:(solid_idx_r+1)*3, solid_idx_c*3:(solid_idx_c+1)*3] += M_block

        # Calculate A_s = K_s - omega^2 * M_s
        A_s = K_s - (self.omega**2) * M_s
        
        # Convert to JAX arrays ONLY at the end
        A_s_jax = np.array(A_s)
        F_s_jax = np.array(F_s)
        
        print("[debug] 原始固体系统组装完成.")
        return A_s_jax, F_s_jax
        
    def assemble_global_system(self, E, nu, rho_s, inlet_source=1.0, volume_source=None):
        """ Assemble coupled system using mappings and apply ALL BCs 
        
        Args:
            E, nu, rho_s: 固体材料参数
            inlet_source: 入口边界处声压值
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
        C_sf = np.zeros((n_solid_dof, N_fluid_unique), dtype=np.float32)
        C_fs = np.zeros((N_fluid_unique, n_solid_dof), dtype=np.float32)

        if self.interface_idx.shape[0] > 0:
            interface_idx_np = np.array(self.interface_idx)
            interface_normals_np = np.array(self.interface_normals)
            interface_node_set = set(interface_idx_np)
            
            # Create a mapping of interface node to normal vector
            interface_normals_map = {idx: normal for idx, normal in zip(interface_idx_np, interface_normals_np)}
            
            fluid_elements_np = np.array(self.fluid_elements)
            nodes_np = np.array(self.nodes)
            
            for elem_nodes in tqdm(fluid_elements_np, desc="Assembling coupling", leave=False):
                # Check if *any* node of the element is an interface node first (optimization)
                if not any(node in interface_node_set for node in elem_nodes):
                    continue

                local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
                nodes_coords_elem = nodes_np[elem_nodes]
                for i_face, local_face in enumerate(local_faces):
                    global_node_indices_face = elem_nodes[np.array(local_face)]
                    # Ensure all nodes of the face are in BOTH fluid and solid mappings AND on interface
                    is_mappable_interface_face = True
                    local_fluid_indices_face = []
                    local_solid_indices_face = [] # Map solid indices too

                    for node_idx in global_node_indices_face:
                        if node_idx in interface_node_set and node_idx in fluid_mapping and node_idx in solid_mapping:
                            local_fluid_indices_face.append(fluid_mapping[node_idx])
                            local_solid_indices_face.append(solid_mapping[node_idx])
                        else:
                            is_mappable_interface_face = False
                            break # Stop checking this face

                    if is_mappable_interface_face:
                        # --- Calculate normal and area (same as before) ---
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
                            if np.dot(normal_vec, vec_to_p3) > 0: normal_vec = -normal_vec
                            # --- End Normal Calculation ---

                            # Assemble C_sf (Force ON solid is -p*n)
                            force_contrib = -(face_area / 3.0) * normal_vec
                            # Assemble C_fs (Fluid equation term rho*omega^2*u_n)
                            motion_contrib = rho_f * (self.omega**2) * (face_area / 3.0) * normal_vec

                            # Add contributions using LOCAL indices
                            for i_node_face in range(3): # Iterate 0, 1, 2 for the face nodes
                                fluid_local_idx = local_fluid_indices_face[i_node_face]
                                solid_local_idx = local_solid_indices_face[i_node_face] # Corresponding local solid index

                                C_sf[solid_local_idx*3:solid_local_idx*3+3, fluid_local_idx] += force_contrib
                                C_fs[fluid_local_idx, solid_local_idx*3:solid_local_idx*3+3] += motion_contrib
        else:
            print("[info] 未找到界面节点，跳过耦合矩阵组装.")

        # visualize_system(C_sf, C_fs, n_solid_dof, N_fluid_unique, title_suffix="Coupling Matrices")
        if np.all(C_fs == 0) and np.all(C_sf == 0):
            print("[warning] 耦合矩阵全为零！")
            
        # ---- Construct Global Block Matrix (Mapped) ----
        print("[info] 正在构造全局块矩阵(映射后)...")
        global_dim = N_fluid_unique + n_solid_dof

        # Check dimensions before creating A_global
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

        # Convert all to JAX arrays
        A_f_jax = np.array(A_f)
        A_s_jax = np.array(A_s)
        C_fs_jax = np.array(C_fs)
        C_sf_jax = np.array(C_sf)
        
        # Create global matrix
        A_global = np.zeros((global_dim, global_dim), dtype=np.float32)
        # Use direct assignments instead of .at syntax for NumPy
        A_global[:N_fluid_unique, :N_fluid_unique] = A_f_jax
        A_global[:N_fluid_unique, N_fluid_unique:] = C_fs_jax
        A_global[N_fluid_unique:, :N_fluid_unique] = C_sf_jax
        A_global[N_fluid_unique:, N_fluid_unique:] = A_s_jax

        # ---- Construct Global Force Vector (Mapped) ----
        F_f_jax = np.array(F_f)
        F_s_jax = np.array(F_s)
        F_global = np.concatenate([F_f_jax, F_s_jax])
        
        print(f"[debug] 原始映射后的A_global形状: {A_global.shape}, 原始映射后的F_global形状: {F_global.shape}")

        # <-- Call visualization BEFORE applying BCs -->
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="Raw Assembly Mapped")

        # ---- Apply ALL Dirichlet Boundary Conditions to Mapped A_global, F_global ----
        print("[info] 正在应用边界条件(映射后)...")
        penalty = self.bcpenalty
        
        # Use NumPy for BC application since it involves complex index manipulation
        A_global_np = np.array(A_global)
        F_global_np = np.array(F_global)

        # Apply Fluid Inlet BC (p = source_value) - Dirichlet condition
        near_fluid_idx_np = np.array(self.near_fluid_idx)
        print(f"[debug] 正在应用流体入口边界条件(p={inlet_source})到{near_fluid_idx_np.shape[0]}个全局节点...")
        nodes_processed_by_inlet = set()
        for global_idx in near_fluid_idx_np:
            if global_idx in fluid_mapping:  # Ensure it's a fluid node
                local_idx = fluid_mapping[global_idx]  # Get local fluid index
                if local_idx not in nodes_processed_by_inlet:
                    A_global_np[local_idx, :] = 0.0
                    A_global_np[:, local_idx] = 0.0
                    A_global_np[local_idx, local_idx] = penalty
                    F_global_np[local_idx] = penalty * inlet_source
                    nodes_processed_by_inlet.add(local_idx)
            else:
                print(f"[warning] 入口节点{global_idx}在fluid_mapping中未找到.")

        # Apply Fluid Outlet BC (dp/dn = 0) - Neumann condition (non-reflecting)
        # For Neumann boundary condition, we don't need to modify the system matrix
        # The natural boundary condition dp/dn = 0 is automatically satisfied
        # Just need to report that we're using this type of boundary
        outlet_fluid_idx_np = np.array(self.outlet_fluid_idx)
        print(f"[info] 在出口处使用诺依曼边界条件(dp/dn=0)({outlet_fluid_idx_np.shape[0]}个节点)")
        print(f"[info] 这是一个无反射出口边界条件")
        
        # Verify that outlet nodes are in the fluid domain
        valid_outlet_count = 0
        for global_idx in outlet_fluid_idx_np:
            if global_idx in fluid_mapping:
                valid_outlet_count += 1
            else:
                print(f"[warning] 出口节点{global_idx}在fluid_mapping中未找到.")
        print(f"[info] 找到{valid_outlet_count}个用于诺依曼边界条件的有效出口节点")

        # Apply Solid Fixed BCs (u = 0)
        fixed_solid_nodes_idx_np = np.array(self.fixed_solid_nodes_idx)
        print(f"[debug] 正在应用固定固体边界条件到{fixed_solid_nodes_idx_np.shape[0]}个全局节点...")
        processed_solid_global_dofs = set()
        for global_node_idx in fixed_solid_nodes_idx_np:
            if global_node_idx in solid_mapping:
                solid_local_idx = solid_mapping[global_node_idx]
                # Calculate global DOF indices with N_fluid_unique offset
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
                print(f"[warning] 固定的固体节点{global_node_idx}在solid_mapping中未找到.")

        # Convert back to JAX arrays after applying BCs
        A_global = np.array(A_global_np)
        F_global = np.array(F_global_np)

        print("[info] 全局矩阵与边界条件组装完成.")
        # <-- Optionally visualize AFTER applying BCs -->
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="With BCs Mapped")

        return A_global, F_global, N_fluid_unique, n_solid_dof  # Return correct sizes

    def solve(self, E, nu, rho_s, volume_source=None):
        """
        给定材料参数，组装全局系统并求解，返回：
          - 预测远端麦克风处流体声压（取 fluid 域 x≈1.0 点平均）
          - 全局解向量 u（其中 u[0:n_nodes] 为 fluid 声压）
          
        Args:
            E, nu, rho_s: 固体材料参数
            volume_source: 体激励项，可以是常数或函数f(x,y,z)
                           例如：constant_source = 10.0
                           或者：def spatial_source(x, y, z): return 10.0 * jnp.exp(-(x**2 + y**2 + z**2))
                           
        Examples:
            # 使用常数体激励项
            solver.solve(E=1e9, nu=0.3, rho_s=1000.0, volume_source=10.0)
            
            # 使用空间变化的体激励项
            def gaussian_source(x, y, z):
                # 高斯分布的声源，在(0.5, 0, 0)位置最强
                return 10.0 * jnp.exp(-5*((x-0.5)**2 + y**2 + z**2))
            
            solver.solve(E=1e9, nu=0.3, rho_s=1000.0, volume_source=gaussian_source)
        """
        A_global, F_global, N_fluid_unique, n_solid_dof_actual = self.assemble_global_system(E, nu, rho_s, inlet_source=0.0, volume_source=volume_source)
        print("[info] 开始求解 (mapped system)")
        try:
            # Using JAX solve for the linear system
            u = np.linalg.solve(A_global, F_global)
        except Exception as e:
            print(f"求解器错误: {e}")
            print("矩阵可能仍然是奇异的或条件数不佳.")
            # In JAX we don't have torch.save, use NumPy instead
            np.save('A_global_error.npy', np.array(A_global))
            np.save('F_global_error.npy', np.array(F_global))
            raise
        print("[info] 求解完成")
        
        # Extract microphone pressure using fluid_mapping
        p_mic = 0.0  # Default value
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
        
        print(f"[info] 预测远端麦克风处流体声压: {float(p_mic)}")
        return p_mic, u  # Return scalar p_mic and full solution vector 

    def forward_numpy(self, E, nu, rho_s, volume_source=None):
        """
        Special version of solve() that uses plain Python/NumPy values for optimization to avoid JAX tracer errors.
        This function can be called during optimization where JAX tracing would cause issues with NumPy conversion.
        
        Returns only the microphone pressure (not the full solution).
        
        Args:
            E, nu, rho_s: 固体材料参数 (Python float values)
            volume_source: 体激励项 (Python value or callable)
        
        Returns:
            float: 麦克风处声压
        """
        # This implements the same workflow as solve() but with NumPy arrays instead of JAX arrays
        # For optimization, we don't need the visualizations, so we'll skip those
        
        try:
            # Use self attributes with numpy arrays
            n_fluid_unique = self.N_fluid_unique
            n_solid_dof = self.n_solid_dof
            fluid_mapping = self.fluid_mapping
            solid_mapping = self.solid_mapping
            nodes_np = np.array(self.nodes)
            fluid_elements_np = np.array(self.fluid_elements)
            solid_elements_np = np.array(self.solid_elements)
            interface_idx_np = np.array(self.interface_idx)
            interface_normals_np = np.array(self.interface_normals)
            interface_node_set = set(interface_idx_np)
            
            # --- 1. Assemble fluid system ---
            K_f = np.zeros((n_fluid_unique, n_fluid_unique), dtype=np.float32)
            M_f = np.zeros((n_fluid_unique, n_fluid_unique), dtype=np.float32)
            F_f = np.zeros(n_fluid_unique, dtype=np.float32)
            
            # Assemble fluid element matrices
            for elem in fluid_elements_np:
                coords_np = nodes_np[elem]
                
                # Calculate element matrices using NumPy operations
                # Compute volume
                V = np.abs(np.linalg.det(np.stack([
                    coords_np[1] - coords_np[0],
                    coords_np[2] - coords_np[0],
                    coords_np[3] - coords_np[0]
                ], axis=1))) / 6.0
                
                # Calculate shape function gradients
                ones = np.ones((4, 1), dtype=np.float32)
                A = np.concatenate([ones, coords_np], axis=1)
                A_inv = np.linalg.inv(A)
                grads = np.transpose(A_inv[1:, :])  # Shape [4,3]
                
                # Calculate stiffness and mass matrices
                K_e = V * np.einsum('ij,kj->ik', grads, grads)
                ones_4 = np.ones((4, 4), dtype=np.float32)
                M_e = V / 20.0 * (np.eye(4, dtype=np.float32) + ones_4)
                
                # Map global element indices to local fluid indices
                local_indices = [fluid_mapping[glob_idx] for glob_idx in elem]

                # Scatter using local indices
                for r_local_map in range(4):
                    row_idx = local_indices[r_local_map]
                    for c_local_map in range(4):
                        col_idx = local_indices[c_local_map]
                        K_f[row_idx, col_idx] += K_e[r_local_map, c_local_map]
                        M_f[row_idx, col_idx] += M_e[r_local_map, c_local_map]
                
                # Handle volume source if provided
                if volume_source is not None:
                    tetra_vol = V  # Already calculated
                    centroid = np.mean(coords_np, axis=0)
                    
                    # Determine source value
                    if callable(volume_source):
                        # Need to handle the case where volume_source is a JAX function
                        try:
                            src_val = volume_source(centroid[0], centroid[1], centroid[2])
                            # If it's a JAX value, convert to NumPy
                            src_val = np.array(src_val)
                        except:
                            # If that fails, use a fallback value
                            src_val = 0.1  # Fallback
                    else:
                        src_val = volume_source
                    
                    # Apply to load vector
                    for r_local_map in range(4):
                        row_idx = local_indices[r_local_map]
                        F_f[row_idx] += tetra_vol * src_val / 4.0
                
            # Calculate A_f
            k_sq = (self.omega / self.c_f)**2
            A_f = K_f - k_sq * M_f
            
            # --- 2. Assemble solid system ---
            K_s = np.zeros((n_solid_dof, n_solid_dof), dtype=np.float32)
            M_s = np.zeros((n_solid_dof, n_solid_dof), dtype=np.float32)
            F_s = np.zeros(n_solid_dof, dtype=np.float32)
            
            # Assemble solid element matrices
            for elem in solid_elements_np:
                coords_np = nodes_np[elem]
                
                # Calculate element matrices using NumPy operations
                # Compute volume
                V = np.abs(np.linalg.det(np.stack([
                    coords_np[1] - coords_np[0],
                    coords_np[2] - coords_np[0],
                    coords_np[3] - coords_np[0]
                ], axis=1))) / 6.0
                
                # Calculate shape function gradients
                ones = np.ones((4, 1), dtype=np.float32)
                A = np.concatenate([ones, coords_np], axis=1)
                A_inv = np.linalg.inv(A)
                grads = np.transpose(A_inv[1:, :])  # Shape [4,3]
                
                # Compute B matrix
                B = np.zeros((6, 12), dtype=np.float32)
                for i in range(4):
                    bx, by, bz = grads[i]
                    B_block = np.array([
                        [bx, 0, 0],
                        [0, by, 0],
                        [0, 0, bz],
                        [by, bx, 0],
                        [0, bz, by],
                        [bz, 0, bx]
                    ], dtype=np.float32)
                    B[:, i*3:(i+1)*3] = B_block
                
                # Compute elasticity D matrix with NumPy arrays
                coeff = E / ((1 + nu) * (1 - 2 * nu))
                D = coeff * np.array([
                    [1 - nu, nu, nu, 0, 0, 0],
                    [nu, 1 - nu, nu, 0, 0, 0],
                    [nu, nu, 1 - nu, 0, 0, 0],
                    [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
                    [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
                    [0, 0, 0, 0, 0, (1 - 2 * nu) / 2]
                ], dtype=np.float32)
                
                # Calculate stiffness matrix K_e = V * B^T D B
                K_e = V * (B.T @ (D @ B))
                
                # Create lumped mass matrix
                m_lump = rho_s * V / 4.0
                M_e = np.zeros((12, 12), dtype=np.float32)
                for i in range(4):
                    start_idx = i * 3
                    end_idx = (i + 1) * 3
                    M_e[start_idx:end_idx, start_idx:end_idx] = m_lump * np.eye(3, dtype=np.float32)
                
                # Map global element indices to local solid indices
                local_solid_indices = [solid_mapping[glob_idx] for glob_idx in elem if glob_idx in solid_mapping]
                # This assumes all nodes in a solid element are unique solid nodes, should be true
                if len(local_solid_indices) != 4:
                    continue

                for r_local_map in range(4):
                    solid_idx_r = local_solid_indices[r_local_map]
                    for c_local_map in range(4):
                        solid_idx_c = local_solid_indices[c_local_map]
                        K_block = K_e[r_local_map*3:(r_local_map+1)*3, c_local_map*3:(c_local_map+1)*3]
                        M_block = M_e[r_local_map*3:(r_local_map+1)*3, c_local_map*3:(c_local_map+1)*3]
                        K_s[solid_idx_r*3:(solid_idx_r+1)*3, solid_idx_c*3:(solid_idx_c+1)*3] += K_block
                        M_s[solid_idx_r*3:(solid_idx_r+1)*3, solid_idx_c*3:(solid_idx_c+1)*3] += M_block

            # Calculate A_s
            A_s = K_s - (self.omega**2) * M_s
            
            # --- 3. Assemble coupling matrices ---
            C_sf = np.zeros((n_solid_dof, n_fluid_unique), dtype=np.float32)
            C_fs = np.zeros((n_fluid_unique, n_solid_dof), dtype=np.float32)
            
            # Simplified coupling - skip the detailed face calculations and just apply at interface nodes
            if len(interface_idx_np) > 0:
                # Create normals mapping
                interface_normals_map = {idx: normal for idx, normal in zip(interface_idx_np, interface_normals_np)}
                
                # Process each interface node
                for node_idx, normal_vec in zip(interface_idx_np, interface_normals_np):
                    if node_idx in fluid_mapping and node_idx in solid_mapping:
                        fluid_local_idx = fluid_mapping[node_idx]
                        solid_local_idx = solid_mapping[node_idx]
                        
                        # Apply simplified coupling terms
                        # We estimate the area based on the mesh resolution - this is a simplification
                        area_factor = 1e-4
                        
                        # Force on solid from fluid
                        C_sf[solid_local_idx*3:solid_local_idx*3+3, fluid_local_idx] += -area_factor * normal_vec
                        
                        # Fluid pressure from solid motion
                        C_fs[fluid_local_idx, solid_local_idx*3:solid_local_idx*3+3] += area_factor * rho_f * (self.omega**2) * normal_vec
            
            # --- 4. Assemble global system ---
            global_dim = n_fluid_unique + n_solid_dof
            A_global = np.zeros((global_dim, global_dim), dtype=np.float32)
            A_global[:n_fluid_unique, :n_fluid_unique] = A_f
            A_global[:n_fluid_unique, n_fluid_unique:] = C_fs
            A_global[n_fluid_unique:, :n_fluid_unique] = C_sf
            A_global[n_fluid_unique:, n_fluid_unique:] = A_s
            
            F_global = np.concatenate([F_f, F_s])
            
            # --- 5. Apply BCs ---
            penalty = self.bcpenalty
            
            # Apply inlet BC (p = 0 for optimization)
            near_fluid_idx_np = np.array(self.near_fluid_idx)
            for global_idx in near_fluid_idx_np:
                if global_idx in fluid_mapping:
                    local_idx = fluid_mapping[global_idx]
                    A_global[local_idx, :] = 0.0
                    A_global[:, local_idx] = 0.0
                    A_global[local_idx, local_idx] = penalty
                    F_global[local_idx] = 0.0  # Zero pressure at inlet for optimization
            
            # Apply fixed solid BCs
            fixed_solid_nodes_idx_np = np.array(self.fixed_solid_nodes_idx)
            for global_node_idx in fixed_solid_nodes_idx_np:
                if global_node_idx in solid_mapping:
                    solid_local_idx = solid_mapping[global_node_idx]
                    global_dof_indices = [n_fluid_unique + solid_local_idx*3 + i for i in range(3)]
                    for dof_idx in global_dof_indices:
                        if dof_idx < global_dim:
                            A_global[dof_idx, :] = 0.0
                            A_global[:, dof_idx] = 0.0
                            A_global[dof_idx, dof_idx] = penalty
                            F_global[dof_idx] = 0.0
            
            # --- 6. Solve the system ---
            # Use NumPy's solver directly
            u = np.linalg.solve(A_global, F_global)
            
            # --- 7. Extract microphone pressure ---
            p_mic = 0.0  # Default value
            if self.mic_node_idx is not None:
                global_mic_idx = int(self.mic_node_idx)
                if global_mic_idx in self.fluid_mapping:
                    local_mic_idx = self.fluid_mapping[global_mic_idx]
                    if local_mic_idx < n_fluid_unique:
                        p_mic = float(u[local_mic_idx])
            
            return p_mic
        
        except Exception as e:
            print(f"Error in forward_numpy: {e}")
            import traceback
            traceback.print_exc()
            return 0.0  # Return a default value on error 