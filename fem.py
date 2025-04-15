# fem.py
import torch
import torch.nn as nn
import numpy as np
import meshio
from tqdm import tqdm
import pyvista as pv
import matplotlib.pyplot as plt
import scipy.sparse as sps
import matplotlib.colors as mcolors # For LogNorm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
def compute_tetra_volume(coords):
    """
    计算四面体体积
    coords: [4,3] 每行为节点坐标
    Volume = abs(det([x2-x1, x3-x1, x4-x1]))/6
    """
    v0 = coords[1] - coords[0]
    v1 = coords[2] - coords[0]
    v2 = coords[3] - coords[0]
    vol = torch.abs(torch.det(torch.stack([v0, v1, v2], dim=1))) / 6.0
    return vol

def compute_shape_function_gradients(coords):
    """
    计算四面体线性形函数梯度（常数）。
    方法：对齐次线性系统 [1 x y z] 的逆矩阵的后三行。
    返回: [4,3]，每一行为对应节点形函数在 x,y,z 方向的梯度。
    """
    ones = torch.ones((4,1), dtype=coords.dtype, device=coords.device)
    A = torch.cat([ones, coords], dim=1)  # [4,4]
    A_inv = torch.inverse(A)
    # 后三行为 b, c, d 系数，对应梯度
    grads = A_inv[1:,:].transpose(0,1)   # shape [4,3]
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
    K_e = V * (grads @ grads.transpose(0,1))
    ones_4 = torch.ones((4,4), dtype=coords.dtype, device=coords.device)
    M_e = V / 20.0 * (torch.eye(4, dtype=coords.dtype, device=coords.device) + ones_4)
    return K_e, M_e

def elasticity_D_matrix(E, nu):
    """
    3D各向同性弹性材料 D 矩阵 (6x6)，Voigt 表示。
    """
    coeff = E / ((1 + nu) * (1 - 2 * nu))
    D = coeff * torch.tensor([
        [1 - nu,    nu,    nu,       0,       0,       0],
        [nu,    1 - nu,    nu,       0,       0,       0],
        [nu,      nu,  1 - nu,       0,       0,       0],
        [0,       0,      0,  (1 - 2 * nu) / 2,  0,       0],
        [0,       0,      0,       0,  (1 - 2 * nu) / 2,  0],
        [0,       0,      0,       0,       0,  (1 - 2 * nu) / 2]
    ], dtype=torch.float32, device=E.device)
    return D

def compute_B_matrix(coords):
    """
    构造四面体单元的应变-位移矩阵 B (6 x 12)。
    利用线性形函数梯度计算，各节点 3 dof。
    """
    grads = compute_shape_function_gradients(coords)  # [4,3]
    B = torch.zeros((6, 12), dtype=coords.dtype, device=coords.device)
    for i in range(4):
        bx, by, bz = grads[i]
        B[:, i*3:(i+1)*3] = torch.tensor([
            [bx,    0,    0],
            [0,   by,     0],
            [0,    0,    bz],
            [by,   bx,    0],
            [0,    bz,   by],
            [bz,   0,    bx]
        ], dtype=coords.dtype, device=coords.device)
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
    K_e = V * (B.transpose(0,1) @ (D @ B))
    # 质量矩阵采用对角 lumped mass:
    M_e = torch.zeros((12, 12), dtype=coords.dtype, device=coords.device)
    m_lump = rho_s * V / 4.0
    for i in range(4):
        M_e[i*3:(i+1)*3, i*3:(i+1)*3] = m_lump * torch.eye(3, dtype=coords.dtype, device=coords.device)
    return K_e, M_e

def visualize_system(A, f, n_nodes, n_solid_dof, title_suffix="Raw System"):
    """
    Visualizes the matrix A using colored scatter plot (log scale) and vector f as a heatmap.
    Marks the divisions between fluid and solid DOFs.
    """
    print(f"[Visualizing System] Matrix shape: {A.shape}, Vector shape: {f.shape}")
    print(f"[Visualizing System] Fluid DOFs (n_nodes): {n_nodes}, Solid DOFs: {n_solid_dof}")
    total_dof = n_nodes + n_solid_dof
    if A.shape[0] != total_dof or A.shape[1] != total_dof or f.shape[0] != total_dof:
        print(f"[Warning] Dimension mismatch in visualize_system: A={A.shape}, f={f.shape}, expected total DOF={total_dof}")
        actual_total_dof = A.shape[0]
        if n_nodes > actual_total_dof: n_nodes = actual_total_dof
    else:
        actual_total_dof = total_dof

    # Move data to CPU
    A_cpu = A.detach().cpu()
    f_cpu = f.detach().cpu().numpy()

    # Convert dense torch tensor to scipy sparse COO matrix for easier access to values/coords
    A_sparse_coo = sps.coo_matrix(A_cpu)
    
    # Filter out small values based on tolerance
    tol = 1e-9
    mask = np.abs(A_sparse_coo.data) > tol
    row = A_sparse_coo.row[mask]
    col = A_sparse_coo.col[mask]
    data = A_sparse_coo.data[mask]
    
    num_non_zero = len(data)
    print(f"[Visualizing System] Matrix sparsity: {num_non_zero / (A.shape[0] * A.shape[1]) * 100:.4f}% non-zero elements > {tol}.")
    
    if num_non_zero == 0:
        print("[Warning] No non-zero elements found in matrix A (above tolerance). Skipping plot.")
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
        
        # Optional block labels inside plot
        # ax.text(n_nodes / 2, n_nodes / 2, 'A_f', va='center', ha='center', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, pad=0.1))
        # ax.text(n_nodes / 2, n_nodes + n_solid_dof / 2, 'C_sf', va='center', ha='center', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, pad=0.1))
        # ax.text(n_nodes + n_solid_dof / 2, n_nodes / 2, 'C_fs', va='center', ha='center', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, pad=0.1))
        # ax.text(n_nodes + n_solid_dof / 2, n_nodes + n_solid_dof / 2, 'A_s', va='center', ha='center', color='white', fontsize=8, bbox=dict(facecolor='black', alpha=0.5, pad=0.1))

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
    print(f"[Visualizing System] Plot saved to {save_path}")
    plt.close(fig)

# ===============================
# Coupled FEM 求解器（全局组装、求解及后处理）
# ===============================
class CoupledFEMSolver(nn.Module):
    def __init__(self, mesh_file, frequency=1000.0, cppenalty=1e8, bcpenalty=1e6):
        """
        mesh_file: "y_pipe.msh" 路径
        frequency: 噪音源频率 (Hz)
        penalty: 流固耦合惩罚参数 β
        """
        super(CoupledFEMSolver, self).__init__()
        self.freq = frequency
        self.omega = 2 * np.pi * frequency
        self.c_f = 343.0
        self.cppenalty = cppenalty
        self.bcpenalty = bcpenalty
        

        # 读网格文件（使用 meshio ）
        print(f"[info] 正在读取网格文件: {mesh_file}")
        mesh = meshio.read(mesh_file)
        # 提取节点（3D 坐标）
        self.nodes = torch.tensor(mesh.points[:, :3], dtype=torch.float32, device=device)
        
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
        
        self.fluid_elements = torch.tensor(fluid_elems_np, dtype=torch.long, device=device)
        self.solid_elements = torch.tensor(solid_elems_np, dtype=torch.long, device=device)

        print(f"[info] 加载了 {self.fluid_elements.shape[0]} 个流体单元 (tag {fluid_tag}) 和 {self.solid_elements.shape[0]} 个固体单元 (tag {solid_tag}).")

        # --- 创建节点映射 ---
        self.fluid_unique_nodes, fluid_inverse_indices = torch.unique(self.fluid_elements.flatten(), return_inverse=True)
        self.solid_unique_nodes, solid_inverse_indices = torch.unique(self.solid_elements.flatten(), return_inverse=True)

        self.N_fluid_unique = len(self.fluid_unique_nodes)
        self.N_solid_unique = len(self.solid_unique_nodes)
        self.n_solid_dof = self.N_solid_unique * 3

        # 全局索引 -> 局部索引映射
        self.fluid_mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(self.fluid_unique_nodes)}
        self.solid_mapping = {global_idx.item(): local_idx for local_idx, global_idx in enumerate(self.solid_unique_nodes)}

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
            
            # 转换为tensor
            self.interface_idx = torch.tensor(list(interface_nodes_set), dtype=torch.long, device=device)
            print(f"[info] 从界面物理组找到 {len(interface_nodes_set)} 个流固界面节点")
        else:
            # 如果找不到界面物理组，退回到找流体和固体共享的节点
            print("[info] 未找到界面物理组，退回到通过流体和固体共享节点识别界面...")
            fluid_node_set = set(self.fluid_unique_nodes.cpu().numpy())
            solid_node_set = set(self.solid_unique_nodes.cpu().numpy())
            interface_node_set = fluid_node_set.intersection(solid_node_set)
            self.interface_idx = torch.tensor(list(interface_node_set), dtype=torch.long, device=device)
            print(f"[info] 找到 {len(interface_node_set)} 个流固界面共享节点")

        # --- 计算界面法向量 ---
        print("[info] 计算界面法向量...")
        node_to_face_normals = {node_id.item(): [] for node_id in self.interface_idx}

        # 迭代流体单元找到界面面及其法向量
        for elem_nodes in tqdm(self.fluid_elements, desc="计算界面法向量"):
            nodes_coords = self.nodes[elem_nodes]  # 4个节点坐标
            # 定义四个面（局部索引）
            local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]

            for i_face, local_face in enumerate(local_faces):
                global_node_indices = elem_nodes[torch.tensor(local_face, device=elem_nodes.device)] # 面的全局节点索引

                # 检查是否所有3个节点都是界面节点
                is_interface_face = all(node_id.item() in node_to_face_normals for node_id in global_node_indices)

                if is_interface_face:
                    # 计算面法向量
                    p0, p1, p2 = self.nodes[global_node_indices]
                    v1 = p1 - p0
                    v2 = p2 - p0
                    normal = torch.cross(v1, v2)
                    norm_mag = torch.norm(normal)

                    if norm_mag > 1e-12: # 避免除以零
                        normal = normal / norm_mag

                        # 确保法向量指向流体元素外部
                        local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                        p3 = nodes_coords[local_idx_p3]
                        face_centroid = (p0 + p1 + p2) / 3.0
                        vec_to_p3 = p3 - face_centroid

                        # 如果法向量指向p3（内部），翻转它
                        if torch.dot(normal, vec_to_p3) > 0:
                            normal = -normal

                        # 将面法向量添加到其3个顶点的列表中
                        for node_id_tensor in global_node_indices:
                            node_id = node_id_tensor.item()
                            if node_id in node_to_face_normals:
                                node_to_face_normals[node_id].append(normal)

        # 通过平均计算最终顶点法向量
        final_normals_list = []
        zero_normal_count = 0
        for node_id_tensor in tqdm(self.interface_idx, desc="平均节点法向量"):
            node_id = node_id_tensor.item()
            normals_to_average = node_to_face_normals.get(node_id, [])

            if not normals_to_average:
                avg_normal = torch.zeros(3, device=device)
                zero_normal_count += 1
            else:
                avg_normal = torch.sum(torch.stack(normals_to_average), dim=0)
                norm_val = torch.norm(avg_normal)
                if norm_val < 1e-12:
                    avg_normal = torch.zeros(3, device=device)
                    zero_normal_count += 1
                else:
                    avg_normal = avg_normal / norm_val
            final_normals_list.append(avg_normal)

        if zero_normal_count > 0:
            print(f"[warning] {zero_normal_count} 个界面节点的法向量为零.")
            
        self.interface_normals = torch.stack(final_normals_list, dim=0)
        
        # 过滤掉法向量为零的节点
        normal_magnitudes = torch.linalg.norm(self.interface_normals, dim=1)
        norm_tolerance = 1e-9
        valid_normal_mask = normal_magnitudes > norm_tolerance
        
        original_count = self.interface_idx.shape[0]
        self.interface_idx = self.interface_idx[valid_normal_mask]
        self.interface_normals = self.interface_normals[valid_normal_mask]
        
        filtered_count = self.interface_idx.shape[0]
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"[info] 移除了 {removed_count} 个法向量无效的界面节点 (norm < {norm_tolerance}).")
        print(f"[info] 最终界面节点数量: {filtered_count}")
        
        # 创建界面节点集合，用于排除它们不被用作其他边界条件
        interface_nodes_set = set(self.interface_idx.cpu().numpy())

        # --- Inlet/Outlet Definitions ---
        # 设置入口条件（x ≈ 0）- 确保它们是流体节点，但不是界面节点
        # Inlet (x approx 0) - Ensure they are fluid nodes but not interface nodes
        potential_near_indices = torch.nonzero(torch.abs(self.nodes[:, 0]) < 1e-3).squeeze()
        near_mask = torch.isin(potential_near_indices, self.fluid_unique_nodes)
        near_fluid_candidates = potential_near_indices[near_mask]
        # 排除界面节点
        non_interface_mask = torch.tensor([idx.item() not in interface_nodes_set for idx in near_fluid_candidates], device=device)
        self.near_fluid_idx = near_fluid_candidates[non_interface_mask]
        print(f"[info] 识别入口节点: 总共 {near_fluid_candidates.shape[0]} 个候选节点, 排除 {near_fluid_candidates.shape[0] - self.near_fluid_idx.shape[0]} 个界面节点, 最终 {self.near_fluid_idx.shape[0]} 个入口节点")

        # 设置主管道出口条件（x ≈ length_main）- 确保它们是流体节点，但不是界面节点
        # Main Outlet (x approx length_main) - Ensure they are fluid nodes but not interface nodes
        outlet_tolerance = 1e-3
        potential_main_outlet_indices = torch.nonzero(torch.abs(self.nodes[:, 0] - length_main) < outlet_tolerance).squeeze()
        main_outlet_mask = torch.isin(potential_main_outlet_indices, self.fluid_unique_nodes)
        main_outlet_candidates = potential_main_outlet_indices[main_outlet_mask]
        # 排除界面节点
        non_interface_mask = torch.tensor([idx.item() not in interface_nodes_set for idx in main_outlet_candidates], device=device)
        main_outlet_idx = main_outlet_candidates[non_interface_mask]
        print(f"[info] 识别主管道出口节点: 总共 {main_outlet_candidates.shape[0]} 个候选节点, 排除 {main_outlet_candidates.shape[0] - main_outlet_idx.shape[0]} 个界面节点")

        # 设置分支出口条件（在分支末端平面）- 确保它们是流体节点，但不是界面节点
        P_junction = torch.tensor([x_junction, 0.0, 0.0], device=device)
        angle_rad = torch.deg2rad(torch.tensor(180 - angle_deg, device=device))
        V_branch_axis = torch.tensor([torch.cos(angle_rad), torch.sin(angle_rad), 0.0], device=device)
        
        P_branch_end = P_junction + length_branch * V_branch_axis
        # 距离分支末端平面近的节点: abs(dot(P - P_branch_end, V_branch_axis)) < tol
        dist_to_branch_end_plane = torch.abs(torch.einsum('nd,d->n', self.nodes - P_branch_end, V_branch_axis))
        
        # 节点也需要在分支管道半径内: 到轴线的距离 <= r_inner + tol
        # 计算到分支轴线的垂直距离
        Vec_P_all = self.nodes - P_junction
        proj_dist_branch = torch.einsum('nd,d->n', Vec_P_all, V_branch_axis)
        Vec_proj = proj_dist_branch.unsqueeze(1) * V_branch_axis.unsqueeze(0)
        perp_dist_branch = torch.norm(Vec_P_all - Vec_proj, dim=1)
        
        branch_outlet_mask = (dist_to_branch_end_plane < outlet_tolerance) & \
                             (perp_dist_branch <= r_inner + outlet_tolerance) & \
                             (proj_dist_branch > 0)  # 确保在分支一侧

        potential_branch_outlet_indices = torch.nonzero(branch_outlet_mask).squeeze()
        branch_outlet_fluid_mask = torch.isin(potential_branch_outlet_indices, self.fluid_unique_nodes)
        branch_outlet_candidates = potential_branch_outlet_indices[branch_outlet_fluid_mask]
        # 排除界面节点
        non_interface_mask = torch.tensor([idx.item() not in interface_nodes_set for idx in branch_outlet_candidates], device=device)
        branch_outlet_idx = branch_outlet_candidates[non_interface_mask]
        print(f"[info] 识别分支出口节点: 总共 {branch_outlet_candidates.shape[0]} 个候选节点, 排除 {branch_outlet_candidates.shape[0] - branch_outlet_idx.shape[0]} 个界面节点")
        
        # 合并主管道和分支管道出口
        combined_outlet_idx = torch.cat((main_outlet_idx, branch_outlet_idx))
        self.outlet_fluid_idx = torch.unique(combined_outlet_idx)
        print(f"[info] 最终出口节点总数: {self.outlet_fluid_idx.shape[0]}")
        
        # 定义麦克风节点（在 x=1.0, y=0, z=0 附近的最近流体节点）
        # Define Mic node (closest fluid node near x=1.0, y=0, z=0 - may need adjustment)
        mic_target_pos = torch.tensor([1.0, 0.0, 0.0], device=device)
        # 找远端节点（例如，x > 0.8 * length_main）并且是流体节点
        potential_far_indices = torch.nonzero(self.nodes[:, 0] > 0.8 * length_main).squeeze()
        far_mask = torch.isin(potential_far_indices, self.fluid_unique_nodes)
        self.far_fluid_idx = potential_far_indices[far_mask]
        
        # 在远端流体节点中找最近的
        if self.far_fluid_idx.numel() > 0:
             far_nodes_coords = self.nodes[self.far_fluid_idx]
             dists_to_mic = torch.norm(far_nodes_coords - mic_target_pos, dim=1)
             self.mic_node_idx = self.far_fluid_idx[torch.argmin(dists_to_mic)] # 单个节点索引
             print(f"  麦克风节点索引: {self.mic_node_idx}, 坐标: {self.nodes[self.mic_node_idx]}")
        else:
             print("[warning] 未找到适合放置麦克风的远端流体节点.")
             self.mic_node_idx = None # 后续处理

        # --- 识别固定边界条件的固体节点 ---
        solid_node_ids_all = torch.unique(self.solid_elements.flatten())
        solid_coords_all = self.nodes[solid_node_ids_all]
        solid_r_yz = torch.linalg.norm(solid_coords_all[:, 1:3], dim=1)
        
        outer_radius_tol = 1e-3
        end_plane_tol = 1e-3
        
        # 查找靠近 x=0 和靠近 r=r_outer 的固体节点
        fixed_solid_mask = (torch.abs(solid_coords_all[:, 0]) < end_plane_tol) & \
                           (torch.abs(solid_r_yz - r_outer) < outer_radius_tol)
        
        fixed_solid_candidates = solid_node_ids_all[fixed_solid_mask]
        # 排除界面节点
        non_interface_mask = torch.tensor([idx.item() not in interface_nodes_set for idx in fixed_solid_candidates], device=device)
        self.fixed_solid_nodes_idx = fixed_solid_candidates[non_interface_mask]
        print(f"[info] 识别固定固体节点: 总共 {fixed_solid_candidates.shape[0]} 个候选节点, 排除 {fixed_solid_candidates.shape[0] - self.fixed_solid_nodes_idx.shape[0]} 个界面节点, 最终 {self.fixed_solid_nodes_idx.shape[0]} 个固定节点")
        
        if self.fixed_solid_nodes_idx.shape[0] < 3: # 通常需要至少3个非共线点
             print("[warning] 固定的固体节点少于3个。刚体模式可能未完全约束.")

        print(f"[info] 初始化完成, fluid elems: {self.fluid_elements.shape[0]}, solid elems: {self.solid_elements.shape[0]}, interface nodes: {self.interface_idx.shape[0]}, inlet nodes: {self.near_fluid_idx.shape[0]}, combined outlet nodes: {self.outlet_fluid_idx.shape[0]}, fixed solid nodes: {self.fixed_solid_nodes_idx.shape[0]}")
        self.visualize_elements()
        input("[info] 按回车继续...")

    
    # 在 CoupledFEMSolver.__init__ 的末尾处新增可视化方法
    def visualize_elements(self):
        # 将节点转换为 CPU 的 numpy 数组
        nodes_np = self.nodes.detach().cpu().numpy()  # [n_nodes, 3]

        # -----------------------------
        # 可视化 Fluid Domain (Tetrahedra)
        # -----------------------------
        fluid_cells_np = self.fluid_elements.detach().cpu().numpy() # Shape (n_fluid_elements, 4)
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
        solid_cells = self.solid_elements.detach().cpu().numpy()  # 每个单元4个节点
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
        interface_nodes = self.nodes[self.interface_idx].detach().cpu().numpy()
        interface_normals = self.interface_normals.detach().cpu().numpy()  # [n_iface, 3]

        # 构造 PyVista 点云
        interface_points = pv.PolyData(interface_nodes)

        # -----------------------------
        # 可视化 Inlet Nodes
        # -----------------------------
        inlet_nodes = self.nodes[self.near_fluid_idx].detach().cpu().numpy()
        inlet_points = pv.PolyData(inlet_nodes)

        # -----------------------------
        # 可视化 Outlet Nodes
        # -----------------------------
        outlet_nodes = self.nodes[self.outlet_fluid_idx].detach().cpu().numpy()
        outlet_points = pv.PolyData(outlet_nodes)
        
        # -----------------------------
        # 可视化 Fixed Solid Nodes
        # -----------------------------
        fixed_solid_nodes = self.nodes[self.fixed_solid_nodes_idx].detach().cpu().numpy()
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
    
    def assemble_global_system(self, E, nu, rho_s, source_value=1.0):
        """ Assemble coupled system using mappings and apply ALL BCs """
        # Get sizes and mappings from self
        N_fluid_unique = self.N_fluid_unique
        n_solid_dof = self.n_solid_dof
        fluid_mapping = self.fluid_mapping
        solid_mapping = self.solid_mapping

        # ---- Raw System Assembly ----
        print("[info] Assembling raw fluid system (mapped)...")
        A_f, F_f = self.assemble_global_fluid_system() # Gets raw mapped A_f, F_f
        # u = torch.linalg.solve(A_f, F_f)
        print("[info] Assembling raw solid system (mapped)...")
        A_s, F_s = self.assemble_global_solid_system(E, nu, rho_s) # Gets raw mapped A_s, F_s
        # u = torch.linalg.solve(A_s, F_s)
        visualize_system(A_f, F_f, N_fluid_unique, n_solid_dof, title_suffix="Raw Fluid System Mapped")
        visualize_system(A_s, F_s, n_solid_dof, N_fluid_unique, title_suffix="Raw Solid System Mapped")
        # ---- Assemble Coupling Matrices (Using Mappings) ----
        print("[info] Assembling coupling matrices (mapped)...")
        C_sf = torch.zeros((n_solid_dof, N_fluid_unique), dtype=torch.float32, device=device)
        C_fs = torch.zeros((N_fluid_unique, n_solid_dof), dtype=torch.float32, device=device)

        if self.interface_idx.numel() > 0:
            interface_node_set = set(self.interface_idx.cpu().numpy())
            # Keep track of interface normals corresponding to self.interface_idx
            interface_normals_map = {global_idx.item(): normal_vec for global_idx, normal_vec in zip(self.interface_idx, self.interface_normals)}

            for elem_nodes in tqdm(self.fluid_elements, desc="Assembling coupling", leave=False):
                # Check if *any* node of the element is an interface node first (optimization)
                if not any(node.item() in interface_node_set for node in elem_nodes):
                    continue

                local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]
                nodes_coords_elem = self.nodes[elem_nodes]
                for i_face, local_face in enumerate(local_faces):
                    global_node_indices_face = elem_nodes[torch.tensor(local_face, device=elem_nodes.device)]
                    # Ensure all nodes of the face are in BOTH fluid and solid mappings AND on interface
                    is_mappable_interface_face = True
                    local_fluid_indices_face = []
                    local_solid_indices_face = [] # Map solid indices too

                    for node_idx_tensor in global_node_indices_face:
                        node_idx = node_idx_tensor.item()
                        if node_idx in interface_node_set and node_idx in fluid_mapping and node_idx in solid_mapping:
                            local_fluid_indices_face.append(fluid_mapping[node_idx])
                            local_solid_indices_face.append(solid_mapping[node_idx])
                        else:
                            is_mappable_interface_face = False
                            break # Stop checking this face

                    if is_mappable_interface_face:
                        # --- Calculate normal and area (same as before) ---
                        p0_idx, p1_idx, p2_idx = global_node_indices_face
                        p0, p1, p2 = self.nodes[p0_idx], self.nodes[p1_idx], self.nodes[p2_idx]
                        v1, v2 = p1 - p0, p2 - p0
                        normal_vec_cross = torch.cross(v1.float(), v2.float())
                        face_area = torch.norm(normal_vec_cross) / 2.0

                        if face_area > 1e-12:
                            normal_vec = normal_vec_cross / (2.0 * face_area)
                            local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                            p3 = nodes_coords_elem[local_idx_p3]
                            face_centroid = (p0 + p1 + p2) / 3.0
                            vec_to_p3 = p3 - face_centroid
                            if torch.dot(normal_vec, vec_to_p3.float()) > 0: normal_vec = -normal_vec
                            # --- End Normal Calculation ---

                            # Assemble C_sf (Force ON solid is -p*n)
                            force_contrib = -(face_area / 3.0) * normal_vec
                            # Assemble C_fs (Fluid equation term rho*omega^2*u_n)
                            motion_contrib = rho_f * (self.omega**2) * (face_area / 3.0) * normal_vec

                            # Add contributions using LOCAL indices
                            for i_node_face in range(3): # Iterate 0, 1, 2 for the face nodes
                                fluid_local_idx = local_fluid_indices_face[i_node_face]
                                solid_local_idx = local_solid_indices_face[i_node_face] # Corresponding local solid index

                                C_sf[solid_local_idx*3 : solid_local_idx*3+3, fluid_local_idx] += force_contrib.type(C_sf.dtype)
                                C_fs[fluid_local_idx, solid_local_idx*3 : solid_local_idx*3+3] += motion_contrib.type(C_fs.dtype)
        else:
            print("[info] No interface nodes found, skipping coupling matrix assembly.")

        # visualize_system(C_sf, C_fs, n_solid_dof, N_fluid_unique, title_suffix="Coupling Matrices")
        if (C_fs==0).all() and (C_sf==0).all():
            print("[warning] Coupling matrices are all zero!")
        # ---- Construct Global Block Matrix (Mapped) ----
        print("[info] Constructing global block matrix (mapped)...")
        global_dim = N_fluid_unique + n_solid_dof

        # Check dimensions before creating A_global
        if A_f.shape != (N_fluid_unique, N_fluid_unique):
            print(f"[Error] A_f shape mismatch! Expected ({N_fluid_unique},{N_fluid_unique}), got {A_f.shape}")
            raise ValueError("Dimension mismatch for A_f")
        if A_s.shape != (n_solid_dof, n_solid_dof):
            print(f"[Error] A_s shape mismatch! Expected ({n_solid_dof},{n_solid_dof}), got {A_s.shape}")
            raise ValueError("Dimension mismatch for A_s")
        if C_fs.shape != (N_fluid_unique, n_solid_dof):
             print(f"[Error] C_fs shape mismatch! Expected ({N_fluid_unique},{n_solid_dof}), got {C_fs.shape}")
             # Attempt resize (risky) or raise error
             # C_fs = C_fs.resize_(N_fluid_unique, n_solid_dof)
             raise ValueError("Dimension mismatch for C_fs")
        if C_sf.shape != (n_solid_dof, N_fluid_unique):
             print(f"[Error] C_sf shape mismatch! Expected ({n_solid_dof},{N_fluid_unique}), got {C_sf.shape}")
             # C_sf = C_sf.resize_(n_solid_dof, N_fluid_unique)
             raise ValueError("Dimension mismatch for C_sf")

        A_global = torch.zeros((global_dim, global_dim), dtype=torch.float32, device=device)
        A_global[0:N_fluid_unique, 0:N_fluid_unique] = A_f
        A_global[0:N_fluid_unique, N_fluid_unique:] = C_fs
        A_global[N_fluid_unique:, 0:N_fluid_unique] = C_sf
        A_global[N_fluid_unique:, N_fluid_unique:] = A_s


        # ---- Construct Global Force Vector (Mapped) ----
        if F_f.shape[0] != N_fluid_unique: F_f = F_f.resize_(N_fluid_unique) # Risky
        if F_s.shape[0] != n_solid_dof: F_s = F_s.resize_(n_solid_dof)     # Risky
        F_global = torch.cat((F_f, F_s), dim=0)
        print(f"[debug] Raw Mapped A_global shape: {A_global.shape}, Raw Mapped F_global shape: {F_global.shape}")

        # <-- Call visualization BEFORE applying BCs -->
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="Raw Assembly Mapped")

        # ---- Apply ALL Dirichlet Boundary Conditions to Mapped A_global, F_global ----
        print("[info] Applying boundary conditions (mapped)...")
        penalty = self.bcpenalty

        # Apply Fluid Inlet BC (p = source_value) - Dirichlet condition
        print(f"[debug] Applying fluid inlet BC (p={source_value}) to {self.near_fluid_idx.shape[0]} global nodes...")
        nodes_processed_by_inlet = set()
        for global_idx_tensor in self.near_fluid_idx:
             global_idx = global_idx_tensor.item()
             if global_idx in fluid_mapping: # Ensure it's a fluid node
                 local_idx = fluid_mapping[global_idx] # Get local fluid index
                 if local_idx not in nodes_processed_by_inlet:
                     A_global[local_idx, :] = 0.0
                     A_global[:, local_idx] = 0.0
                     A_global[local_idx, local_idx] = penalty
                     F_global[local_idx] = penalty * source_value
                     nodes_processed_by_inlet.add(local_idx)
             else:
                 print(f"[warning] Inlet node {global_idx} not found in fluid_mapping.")

        # Apply Fluid Outlet BC (dp/dn = 0) - Neumann condition (non-reflecting)
        # For Neumann boundary condition, we don't need to modify the system matrix
        # The natural boundary condition dp/dn = 0 is automatically satisfied
        # Just need to report that we're using this type of boundary
        print(f"[info] Using Neumann boundary condition (dp/dn=0) at outlets ({self.outlet_fluid_idx.shape[0]} nodes)")
        print(f"[info] This is a non-reflecting outlet boundary condition")
        
        # Verify that outlet nodes are in the fluid domain
        valid_outlet_count = 0
        for global_idx_tensor in self.outlet_fluid_idx:
            global_idx = global_idx_tensor.item()
            if global_idx in fluid_mapping:
                valid_outlet_count += 1
            else:
                print(f"[warning] Outlet node {global_idx} not found in fluid_mapping.")
        print(f"[info] Found {valid_outlet_count} valid outlet nodes for Neumann boundary condition")

        # Apply Solid Fixed BCs (u = 0)
        print(f"[debug] Applying fixed solid BCs to {self.fixed_solid_nodes_idx.shape[0]} global nodes...")
        processed_solid_global_dofs = set()
        for global_node_idx_tensor in self.fixed_solid_nodes_idx:
            global_node_idx = global_node_idx_tensor.item()
            if global_node_idx in solid_mapping:
                 solid_local_idx = solid_mapping[global_node_idx]
                 # Calculate global DOF indices with N_fluid_unique offset
                 global_dof_indices = [N_fluid_unique + solid_local_idx*3 + i for i in range(3)]

                 for dof_idx in global_dof_indices:
                      if dof_idx < global_dim:
                           if dof_idx not in processed_solid_global_dofs and A_global[dof_idx, dof_idx] != penalty:
                                A_global[dof_idx, :] = 0.0
                                A_global[:, dof_idx] = 0.0
                                A_global[dof_idx, dof_idx] = penalty
                                F_global[dof_idx] = 0.0
                                processed_solid_global_dofs.add(dof_idx)
                      else:
                           print(f"[warning] Calculated solid DOF index {dof_idx} out of bounds ({global_dim}).")
            else:
                 print(f"[warning] Fixed solid node {global_node_idx} not found in solid_mapping.")


        print("[info] Global matrix and BC assembly complete.")
        # <-- Optionally visualize AFTER applying BCs -->
        visualize_system(A_global, F_global, N_fluid_unique, n_solid_dof, title_suffix="With BCs Mapped")

        return A_global, F_global, N_fluid_unique, n_solid_dof # Return correct sizes

    def solve(self, E, nu, rho_s):
        """
        给定材料参数，组装全局系统并求解，返回：
          - 预测远端麦克风处流体声压（取 fluid 域 x≈1.0 点平均）
          - 全局解向量 u（其中 u[0:n_nodes] 为 fluid 声压）
        """
        A_global, F_global, N_fluid_unique, n_solid_dof_actual = self.assemble_global_system(E, nu, rho_s)
        print("[info] 开始求解 (mapped system)")
        try:
            u = torch.linalg.lstsq(A_global, F_global).solution
        except torch._C._LinAlgError as e:
            print(f"Solver Error: {e}")
            print("Matrix might still be singular or ill-conditioned.")
            torch.save(A_global, 'A_global_error.pt')
            torch.save(F_global, 'F_global_error.pt')
            raise
        print("[info] 求解完成")
        print(f"[info] topk: {torch.topk(u, 100)}")
        # Extract microphone pressure using fluid_mapping
        p_mic = torch.tensor(0.0, device=device, dtype=u.dtype) # Default value
        if self.mic_node_idx is not None:
             global_mic_idx = self.mic_node_idx.item()
             if global_mic_idx in self.fluid_mapping:
                  local_mic_idx = self.fluid_mapping[global_mic_idx]
                  if local_mic_idx < N_fluid_unique: # Double check bounds
                       p_mic = u[local_mic_idx]
                  else:
                       print(f"[warning] Mapped mic index {local_mic_idx} out of bounds for fluid DOFs ({N_fluid_unique}).")
             else:
                  print(f"[warning] Global mic node index {global_mic_idx} not found in fluid_mapping.")
        else:
             print("[warning] Mic node index not defined.")
        print(f"[info] 预测远端麦克风处流体声压: {p_mic.squeeze()}")
        exit(1)
        return p_mic.squeeze(), u # Return scalar p_mic

    def assemble_global_fluid_system(self):
        """ Assemble raw fluid system A_f, F_f (Helmholtz) using local fluid mapping """
        # Use pre-calculated sizes and mapping
        n_fluid_local_dof = self.N_fluid_unique # Size based on unique fluid nodes
        fluid_mapping = self.fluid_mapping

        K_f = torch.zeros((n_fluid_local_dof, n_fluid_local_dof), dtype=torch.float32, device=device)
        M_f = torch.zeros((n_fluid_local_dof, n_fluid_local_dof), dtype=torch.float32, device=device)
        F_f = torch.zeros(n_fluid_local_dof, dtype=torch.float32, device=device)

        print("[debug] Assembling fluid K_f and M_f (raw, mapped)...")
        for elem in tqdm(self.fluid_elements, desc="Assembling fluid K/M", leave=False):
            coords = self.nodes[elem]
            K_e, M_e = element_matrices_fluid(coords)
            # Map global element indices to local fluid indices
            local_indices = [fluid_mapping[glob_idx.item()] for glob_idx in elem]

            # Scatter using local indices
            for r_local_map in range(4): # Index within local_indices (0-3)
                 row_idx = local_indices[r_local_map]
                 for c_local_map in range(4):
                      col_idx = local_indices[c_local_map]
                      K_f[row_idx, col_idx] += K_e[r_local_map, c_local_map]
                      M_f[row_idx, col_idx] += M_e[r_local_map, c_local_map]

        k_sq = (self.omega / self.c_f)**2
        A_f = K_f - k_sq * M_f
        print("[debug] Raw fluid system assembly complete.")
        return A_f, F_f

    def assemble_global_solid_system(self, E, nu, rho_s):
        """ Assemble raw solid system A_s, F_s using local solid mapping """
        # Use pre-calculated sizes and mapping
        n_solid_dof = self.n_solid_dof
        solid_mapping = self.solid_mapping

        # No need to calculate mapping here anymore
        # solid_unique_nodes = torch.unique(self.solid_elements.flatten())
        # solid_mapping = {node_id.item(): i for i, node_id in enumerate(solid_unique_nodes)}
        # n_solid_nodes_unique = len(solid_unique_nodes)
        # n_solid_dof = n_solid_nodes_unique * 3

        K_s = torch.zeros((n_solid_dof, n_solid_dof), dtype=torch.float32, device=device)
        M_s = torch.zeros((n_solid_dof, n_solid_dof), dtype=torch.float32, device=device)
        F_s = torch.zeros(n_solid_dof, dtype=torch.float32, device=device)

        print("[debug] Assembling solid K_s and M_s (raw, mapped)...")
        for elem in tqdm(self.solid_elements, desc="Assembling solid K/M", leave=False):
            coords = self.nodes[elem]
            K_e, M_e = element_matrices_solid(coords, E, nu, rho_s)
            # Map global element indices to local solid indices
            local_solid_indices = [solid_mapping[glob_idx.item()] for glob_idx in elem if glob_idx.item() in solid_mapping]
            # This assumes all nodes in a solid element are unique solid nodes, should be true
            if len(local_solid_indices) != 4:
                 print(f"[Warning] Solid element {elem} has nodes not in solid_mapping?")
                 continue

            for r_local_map in range(4): # Index referring to element's node (0-3)
                 solid_idx_r = local_solid_indices[r_local_map] # Local solid index
                 for c_local_map in range(4):
                      solid_idx_c = local_solid_indices[c_local_map]
                      # Get the 3x3 block from K_e and M_e based on the element's node order
                      K_block = K_e[r_local_map*3 : (r_local_map+1)*3, c_local_map*3 : (c_local_map+1)*3]
                      M_block = M_e[r_local_map*3 : (r_local_map+1)*3, c_local_map*3 : (c_local_map+1)*3]
                      # Add to the global solid matrices at the mapped indices
                      K_s[solid_idx_r*3 : (solid_idx_r+1)*3, solid_idx_c*3 : (solid_idx_c+1)*3] += K_block
                      M_s[solid_idx_r*3 : (solid_idx_r+1)*3, solid_idx_c*3 : (solid_idx_c+1)*3] += M_block

        # Calculate A_s = K_s - omega^2 * M_s
        A_s = K_s - (self.omega**2) * M_s
        print("[debug] Raw solid system assembly complete.")
        # Return only A_s, F_s (mapping is now in self)
        return A_s, F_s

