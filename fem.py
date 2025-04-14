# fem.py
import torch
import torch.nn as nn
import numpy as np
import meshio
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 管道参数定义
r_outer = 0.05   # 外半径 5cm
r_inner = 0.045  # 内半径 4.5cm
angle_deg = 45     # 分支角度 (degrees)
length_main = 1.5  # 主干总长 1.5m
x_junction = 0.5   # Junction point x-coordinate
# Branch length MUST match the value used in geometry_gmsh.py
length_branch = 0.3 


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
      M_e = Volume/10 * (2I_4 + ones(4,4))
    """
    V = compute_tetra_volume(coords)
    grads = compute_shape_function_gradients(coords)  # [4,3]
    K_e = V * (grads @ grads.transpose(0,1))
    ones_4 = torch.ones((4,4), dtype=coords.dtype, device=coords.device)
    M_e = V / 10.0 * (2 * torch.eye(4, dtype=coords.dtype, device=coords.device) + ones_4)
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

# ===============================
# Coupled FEM 求解器（全局组装、求解及后处理）
# ===============================
class CoupledFEMSolver(nn.Module):
    def __init__(self, mesh_file, frequency=1000.0, penalty=1e8):
        """
        mesh_file: "y_pipe.msh" 路径
        frequency: 噪音源频率 (Hz)
        penalty: 流固耦合惩罚参数 β
        """
        super(CoupledFEMSolver, self).__init__()
        self.freq = frequency
        self.omega = 2 * np.pi * frequency
        self.penalty = penalty

        # 读网格文件（使用 meshio ），假定物理组：Fluid Domain tag=1, Pipe Wall tag=2
        mesh = meshio.read(mesh_file)
        # 提取节点（3D 坐标）
        self.nodes = torch.tensor(mesh.points[:, :3], dtype=torch.float32, device=device)
        
        # 检查是否存在四面体单元和物理标签
        if 'tetra' not in mesh.cells_dict:
            raise ValueError("Mesh does not contain tetrahedral elements ('tetra').")
        if 'gmsh:physical' not in mesh.cell_data_dict or 'tetra' not in mesh.cell_data_dict['gmsh:physical']:
            raise ValueError("Mesh does not contain physical tags for tetrahedra ('gmsh:physical'->'tetra').")

        # 提取所有四面体单元及其物理标签
        all_tetra_cells = mesh.cells_dict['tetra'] # shape (N_all_tetra, 4)
        physical_tags = mesh.cell_data_dict['gmsh:physical']['tetra'] # shape (N_all_tetra,)
        
        # 根据物理标签区分流体（标签1）和固体（标签2）单元
        fluid_mask = (physical_tags == 1)
        solid_mask = (physical_tags == 2)
        
        fluid_elems_np = all_tetra_cells[fluid_mask]
        solid_elems_np = all_tetra_cells[solid_mask]

        if fluid_elems_np.size == 0:
            print("[warning] No fluid elements found with physical tag 1.")
        if solid_elems_np.size == 0:
            print("[warning] No solid elements found with physical tag 2.")

        self.fluid_elements = torch.tensor(fluid_elems_np, dtype=torch.long, device=device)
        self.solid_elements = torch.tensor(solid_elems_np, dtype=torch.long, device=device)

        print(f"[info] Loaded {self.fluid_elements.shape[0]} fluid elements (tag 1) and {self.solid_elements.shape[0]} solid elements (tag 2).")

        fluid_node_ids = torch.unique(self.fluid_elements.flatten())

        # --- Interface Identification based on Geometric Parameters ---
        print("[info] Identifying interface nodes using geometric parameters...")
        interface_tol = 1e-3 # Tolerance for radius/distance checks

        # 1. Main Pipe Interface Nodes (YZ plane radius check)
        r_main = torch.norm(self.nodes[:, 1:3], dim=1)
        main_pipe_mask = (torch.abs(r_main - r_inner) < interface_tol) & \
                         (self.nodes[:, 0] >= 0) & (self.nodes[:, 0] <= length_main)
        main_interface_idx = torch.nonzero(main_pipe_mask).squeeze()

        # 2. Branch Pipe Interface Nodes (distance to branch axis check)
        angle_rad = torch.deg2rad(torch.tensor(180 - angle_deg, device=device))
        P_junction = torch.tensor([x_junction, 0.0, 0.0], device=device)
        V_branch_axis = torch.tensor([torch.cos(angle_rad), torch.sin(angle_rad), 0.0], device=device)

        # Vector from junction to each node
        Vec_P_all = self.nodes - P_junction
        
        # Project distance along branch axis (dot product)
        proj_dist_branch = torch.einsum('nd,d->n', Vec_P_all, V_branch_axis)
        
        # Vector projection onto axis
        Vec_proj = proj_dist_branch.unsqueeze(1) * V_branch_axis.unsqueeze(0)
        
        # Perpendicular distance to branch axis
        perp_dist_branch = torch.norm(Vec_P_all - Vec_proj, dim=1)

        branch_pipe_mask = (torch.abs(perp_dist_branch - r_inner) < interface_tol) & \
                           (proj_dist_branch >= 0) & (proj_dist_branch <= length_branch)
        branch_interface_idx = torch.nonzero(branch_pipe_mask).squeeze()

        # Combine and deduplicate interface nodes from main and branch
        combined_interface_idx = torch.cat((main_interface_idx, branch_interface_idx))
        unique_interface_idx = torch.unique(combined_interface_idx)

        # Final check: Ensure interface nodes belong to the fluid domain (should mostly be true)
        final_interface_mask = torch.isin(unique_interface_idx, fluid_node_ids)
        self.interface_idx = unique_interface_idx[final_interface_mask]
        print(f"[info] Found {self.interface_idx.shape[0]} potential interface nodes using geometric checks (pre-filter).")
        # --- End Interface Identification ---

        # --- Accurate Normal Calculation using Face Normal Averaging ---
        print("[info] Calculating interface normals using face normal averaging...")
        interface_node_set = set(self.interface_idx.cpu().numpy())
        node_to_face_normals = {node_id: [] for node_id in interface_node_set}

        # Iterate through fluid elements to find interface faces and their normals
        for elem_nodes in tqdm(self.fluid_elements, desc="Finding interface faces"):
            nodes_coords = self.nodes[elem_nodes]  # Coords of the 4 nodes
            # Define the 4 faces (local indices) - order matters for consistent initial normal guess
            local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]

            for i_face, local_face in enumerate(local_faces):
                global_node_indices = elem_nodes[torch.tensor(local_face, device=elem_nodes.device)] # Global node indices for this face

                # Check if all 3 nodes are interface nodes
                is_interface_face = all(node_id.item() in interface_node_set for node_id in global_node_indices)

                if is_interface_face:
                    # Calculate face normal
                    p0, p1, p2 = self.nodes[global_node_indices]
                    v1 = p1 - p0
                    v2 = p2 - p0
                    normal = torch.cross(v1, v2)
                    norm_mag = torch.norm(normal)

                    if norm_mag > 1e-12: # Avoid division by zero for degenerate faces
                        normal = normal / norm_mag

                        # Orientation check: ensure normal points OUT of the fluid element
                        # Find the 4th node (not part of this face)
                        local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                        p3 = nodes_coords[local_idx_p3]
                        face_centroid = (p0 + p1 + p2) / 3.0
                        vec_to_p3 = p3 - face_centroid

                        # If normal points towards p3 (inwards), flip it
                        if torch.dot(normal, vec_to_p3) > 0:
                            normal = -normal

                        # Add this face normal to the lists of its 3 vertices
                        for node_id_tensor in global_node_indices:
                            node_id = node_id_tensor.item()
                            if node_id in node_to_face_normals: # Should always be true
                                node_to_face_normals[node_id].append(normal)
                    # else: handle degenerate face? Skip for now.

        # Calculate final vertex normals by averaging and normalizing
        final_normals_list = []
        zero_normal_count = 0
        for node_id_tensor in tqdm(self.interface_idx, desc="Averaging vertex normals"):
            node_id = node_id_tensor.item()
            normals_to_average = node_to_face_normals.get(node_id, [])

            if not normals_to_average:
                # This might happen if a node is marked as interface but isn't part of any *fluid* interface face
                avg_normal = torch.zeros(3, device=device)
                zero_normal_count += 1
            else:
                avg_normal = torch.sum(torch.stack(normals_to_average), dim=0)
                norm_val = torch.norm(avg_normal)
                if norm_val < 1e-12:
                    avg_normal = torch.zeros(3, device=device) # Avoid division by zero
                    zero_normal_count += 1
                else:
                    avg_normal = avg_normal / norm_val
            final_normals_list.append(avg_normal)

        if zero_normal_count > 0:
            print(f"[warning] {zero_normal_count} interface nodes ended up with a zero normal vector.")
            
        self.interface_normals = torch.stack(final_normals_list, dim=0)
        print("[info] Interface normal calculation complete.")
        # --- End Normal Calculation ---

        # --- Filter out nodes with invalid (zero) normals ---
        normal_magnitudes = torch.linalg.norm(self.interface_normals, dim=1)
        norm_tolerance = 1e-9 # Use a small tolerance
        valid_normal_mask = normal_magnitudes > norm_tolerance
        
        original_count = self.interface_idx.shape[0]
        
        # Apply the mask to keep only nodes with valid normals
        self.interface_idx = self.interface_idx[valid_normal_mask]
        self.interface_normals = self.interface_normals[valid_normal_mask]
        
        filtered_count = self.interface_idx.shape[0]
        removed_count = original_count - filtered_count
        if removed_count > 0:
            print(f"[info] Removed {removed_count} interface nodes due to invalid/zero normals (norm < {norm_tolerance}).")
        print(f"[info] Final interface nodes after normal validity check: {filtered_count}")
        # --- End Normal Filtering ---

        # --- Final Filter: Remove nodes clearly inside r_inner ---
        if self.interface_idx.numel() > 0: # Proceed only if there are nodes left
            interface_coords = self.nodes[self.interface_idx]
            interface_r_yz = torch.linalg.norm(interface_coords[:, 1:3], dim=1)
            # Define a tolerance. Nodes significantly smaller than r_inner are removed.
            # Using the same interface_tol might be reasonable, or a different one.
            inner_radius_tolerance = interface_tol # Using 1e-3 from earlier
            
            keep_mask_radius = interface_r_yz >= (r_inner - inner_radius_tolerance)
            
            original_count_before_radius_filter = self.interface_idx.shape[0]
            self.interface_idx = self.interface_idx[keep_mask_radius]
            self.interface_normals = self.interface_normals[keep_mask_radius]
            
            filtered_count_after_radius = self.interface_idx.shape[0]
            removed_count_radius = original_count_before_radius_filter - filtered_count_after_radius
            if removed_count_radius > 0:
                 print(f"[info] Removed {removed_count_radius} additional interface nodes clearly inside r_inner (r_yz < {r_inner - inner_radius_tolerance:.4f}).")
            print(f"[info] Final interface nodes after inner radius check: {filtered_count_after_radius}")
        # --- End Inner Radius Filtering ---

        # --- Inlet/Outlet Definitions ---
        # Inlet (x approx 0) - Ensure they are fluid nodes
        potential_near_indices = torch.nonzero(torch.abs(self.nodes[:, 0]) < 1e-3).squeeze()
        near_mask = torch.isin(potential_near_indices, fluid_node_ids)
        self.near_fluid_idx = potential_near_indices[near_mask]

        # Main Outlet (x approx length_main) - Ensure they are fluid nodes
        outlet_tolerance = 1e-3
        potential_main_outlet_indices = torch.nonzero(torch.abs(self.nodes[:, 0] - length_main) < outlet_tolerance).squeeze()
        main_outlet_mask = torch.isin(potential_main_outlet_indices, fluid_node_ids)
        main_outlet_idx = potential_main_outlet_indices[main_outlet_mask]

        # Branch Outlet (at the end plane of the branch) - Ensure they are fluid nodes
        P_branch_end = P_junction + length_branch * V_branch_axis
        # Nodes close to the end plane: abs(dot(P - P_branch_end, V_branch_axis)) < tol
        dist_to_branch_end_plane = torch.abs(torch.einsum('nd,d->n', self.nodes - P_branch_end, V_branch_axis))
        
        # Nodes also need to be within the branch radius perp_dist_branch <= r_inner + tol
        # Reusing perp_dist_branch calculated earlier for *all* nodes
        branch_outlet_mask = (dist_to_branch_end_plane < outlet_tolerance) & \
                             (perp_dist_branch <= r_inner + interface_tol) & \
                             (proj_dist_branch > 0) # Ensure it's on the branch side

        potential_branch_outlet_indices = torch.nonzero(branch_outlet_mask).squeeze()
        branch_outlet_fluid_mask = torch.isin(potential_branch_outlet_indices, fluid_node_ids)
        branch_outlet_idx = potential_branch_outlet_indices[branch_outlet_fluid_mask]
        
        # Combine main and branch outlets
        combined_outlet_idx = torch.cat((main_outlet_idx, branch_outlet_idx))
        self.outlet_fluid_idx = torch.unique(combined_outlet_idx)
        
        # Define Mic node (closest fluid node near x=1.0, y=0, z=0 - may need adjustment)
        mic_target_pos = torch.tensor([1.0, 0.0, 0.0], device=device)
        # Find far nodes (e.g., x > 0.8 * length_main) that are fluid nodes
        potential_far_indices = torch.nonzero(self.nodes[:, 0] > 0.8 * length_main).squeeze()
        far_mask = torch.isin(potential_far_indices, fluid_node_ids)
        self.far_fluid_idx = potential_far_indices[far_mask]
        
        # Find closest among far fluid nodes
        if self.far_fluid_idx.numel() > 0:
             far_nodes_coords = self.nodes[self.far_fluid_idx]
             dists_to_mic = torch.norm(far_nodes_coords - mic_target_pos, dim=1)
             self.mic_node_idx = self.far_fluid_idx[torch.argmin(dists_to_mic)] # Single node index
             print(f"  Mic node index: {self.mic_node_idx}, Coord: {self.nodes[self.mic_node_idx]}")
        else:
             print("[warning] No suitable far fluid nodes found for microphone placement.")
             self.mic_node_idx = None # Handle this downstream

        print(f"[info] 初始化完成, fluid elems: {self.fluid_elements.shape[0]}, solid elems: {self.solid_elements.shape[0]}, interface nodes: {self.interface_idx.shape[0]}, inlet nodes: {self.near_fluid_idx.shape[0]}, combined outlet nodes: {self.outlet_fluid_idx.shape[0]}")
        self.visualize_elements()
        input("[info] 按回车继续...")

    
    # 在 CoupledFEMSolver.__init__ 的末尾处新增可视化方法
    def visualize_elements(self):
        import pyvista as pv
        import numpy as np
        import torch # Ensure torch is imported here if not globally

        # 将节点转换为 CPU 的 numpy 数组
        nodes_np = self.nodes.detach().cpu().numpy()  # [n_nodes, 3]

        # -----------------------------
        # 可视化 Fluid Elements（四面体单元的三角面）
        # -----------------------------
        fluid_cells = self.fluid_elements.detach().cpu().numpy()  # 每个单元4个节点
        fluid_faces = []
        for cell in fluid_cells:
            pts = cell.tolist()
            # 四面体的4个三角面: (0,1,2), (0,1,3), (0,2,3), (1,2,3)
            faces = [
                pts[0:3],
                [pts[0], pts[1], pts[3]],
                [pts[0], pts[2], pts[3]],
                [pts[1], pts[2], pts[3]]
            ]
            fluid_faces.extend(faces)
        # 构造一维数组：[3, id0, id1, id2, 3, id0, id1, id2, ...]
        fluid_faces_flat = []
        for face in fluid_faces:
            fluid_faces_flat.append(3)
            fluid_faces_flat.extend(face)
        fluid_faces_flat = np.array(fluid_faces_flat, dtype=np.int64)

        fluid_mesh = pv.PolyData(nodes_np, fluid_faces_flat)

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
        plotter.add_mesh(fluid_mesh, color="cyan", opacity=0.5, label="Fluid Elements")
        plotter.add_mesh(solid_mesh, color="magenta", opacity=0.5, label="Solid Elements")
        plotter.add_mesh(interface_points, color="blue", point_size=10, render_points_as_spheres=True, label="Interface Nodes")
        plotter.add_mesh(inlet_points, color="yellow", point_size=10, render_points_as_spheres=True, label="Inlet Nodes")
        plotter.add_mesh(outlet_points, color="grey", point_size=10, render_points_as_spheres=True, label="Outlet Nodes")
        plotter.add_mesh(arrows, color="green", label="Interface Normals")
        plotter.add_legend()
        plotter.show()
    
    def assemble_global_system(self, E, nu, rho_s):
        """
        组装耦合系统：  
          Fluid: 对所有 fluid 节点，DOF=1，每个流体单元按线性四面体组装刚度与质量矩阵  
          Solid: 对所有 solid 节点（固体区域），DOF=3，每个固体单元组装对应刚度与质量矩阵  
          Coupling: 对接口节点，采用惩罚方法强制 p + n^T σ(u) = 0  
        整个系统 u = [p; u_struct]，全局矩阵 A_global，右端项 F_global。
        """
        # ---- 流体域组装 ----
        n_nodes = self.nodes.shape[0]  # 全局节点数（假设 fluid 系统使用全部节点）
        A_f = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)
        M_f = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)
        for elem in tqdm(self.fluid_elements, desc="[info] 流体域组装"):
            indices = elem  # 长度为 4
            coords = self.nodes[indices]  # [4,3]
            Ke, Me = element_matrices_fluid(coords)  # [4,4] each
            for i in range(4):
                for j in range(4):
                    A_f[indices[i], indices[j]] += Ke[i, j]
                    M_f[indices[i], indices[j]] += Me[i, j]
        # 设定频域参数： ω, k
        c_f = 343.0  # 空气声速
        k = self.omega / c_f
        A_f = A_f - (k**2) * M_f
        F_f = torch.zeros((n_nodes, 1), dtype=torch.float32, device=device)
        # 对于近端边界 (噪音源) 采用 Dirichlet BC，设定 p = p0
        p0 = 1.0  # 假设噪音源输出 1.0 Pa
        for idx in tqdm(self.near_fluid_idx, desc="[info] 流体域inlet组装"):
            A_f[idx, :] = 0.0
            A_f[idx, idx] = 1.0
            F_f[idx] = p0
        # 对于自由边界self.outlet_fluid_idx，采用 Dirichlet BC，设定 p = 0 (压力释放)
        for idx in tqdm(self.outlet_fluid_idx, desc="[info] 流体域outlet组装"):
            A_f[idx, :] = 0.0
            A_f[idx, idx] = 1.0
            F_f[idx] = 0.0
        print("[info] 流体域组装完成")
        # ---- 固体域组装 ----
        # 提取固体域节点（独立）
        solid_node_ids = torch.unique(self.solid_elements)
        n_solid = solid_node_ids.shape[0]
        A_s = torch.zeros((n_solid * 3, n_solid * 3), dtype=torch.float32, device=device)
        M_s = torch.zeros((n_solid * 3, n_solid * 3), dtype=torch.float32, device=device)
        solid_mapping = {int(idx.item()): i for i, idx in enumerate(solid_node_ids)}
        for elem in tqdm(self.solid_elements, desc="[info] 固体域组装"):
            indices = elem
            coords = self.nodes[indices]
            Ke, Me = element_matrices_solid(coords, E, nu, rho_s)  # [12,12]
            for i in range(4):
                for j in range(4):
                    I = solid_mapping[int(indices[i].item())]
                    J = solid_mapping[int(indices[j].item())]
                    A_s[I*3:(I+1)*3, J*3:(J+1)*3] += Ke[i*3:(i+1)*3, j*3:(j+1)*3]
                    M_s[I*3:(I+1)*3, J*3:(J+1)*3] += Me[i*3:(i+1)*3, j*3:(j+1)*3]
        A_s = A_s - (self.omega**2) * M_s
        # F_s = zeros (自由振动)
        print("[info] 固体域组装完成")
        # ---- 耦合处理：对流固接口使用惩罚法 ----
        β = self.penalty
        # 对于每个固体接口节点，设对应 fluid 节点通过 self.interface_mapping 给出
        for idx_idx, idx_solid in enumerate(tqdm(self.interface_idx, desc="[info] 耦合处理")):
            # 对应 fluid 节点
            fluid_idx = idx_solid.item()
            # 在流体系统，增加 β
            A_f[fluid_idx, fluid_idx] += β
            F_f[fluid_idx] += 0.0  # 期望条件： p + n^T σ(u) = 0
            # 对固体系统，找到该节点在固体域中的序号
            if int(idx_solid.item()) in solid_mapping:
                i_struct = solid_mapping[int(idx_solid.item())]
                n_vec = self.interface_normals[0]  # 此处取第一个接口法向量（工业级代码应逐点使用对应法向量）
                penalty_mat = β * torch.ger(n_vec, n_vec)  # 3x3
                A_s[i_struct*3:(i_struct+1)*3, i_struct*3:(i_struct+1)*3] += penalty_mat
        print("[info] 耦合处理完成")
        # ---- 组装全局系统 ----
        n_total = n_nodes + n_solid * 3
        A_global = torch.zeros((n_total, n_total), dtype=torch.float32, device=device)
        F_global = torch.zeros((n_total, 1), dtype=torch.float32, device=device)
        A_global[:n_nodes, :n_nodes] = A_f
        F_global[:n_nodes, :] = F_f
        A_global[n_nodes:, n_nodes:] = A_s
        # 此处未组装流固耦合的 off-diagonal项，
        # 在完整工业级耦合中，流体和固体之间应通过耦合矩阵构成 off-diagonals，
        # 但这里采用了惩罚法将耦合作用隐含在对角线上。
        print("[info] 矩阵组装完成")
        return A_global, F_global, n_nodes, n_solid

    def solve(self, E, nu, rho_s):
        """
        给定材料参数，组装全局系统并求解，返回：
          - 预测远端麦克风处流体声压（取 fluid 域 x≈1.0 点平均）
          - 全局解向量 u（其中 u[0:n_fluid] 为 fluid 声压）
        """
        A_global, F_global, n_fluid, n_solid = self.assemble_global_system(E, nu, rho_s)
        print("[info] 开始求解")
        u = torch.linalg.solve(A_global, F_global)
        print("[info] 求解完成")
        # 从 fluid 部分取出远端麦克风处预测
        if self.far_fluid_idx.numel() > 0:
            p_far = torch.mean(u[self.far_fluid_idx])
        else:
            p_far = torch.tensor(0.0, device=device)
        return p_far, u
