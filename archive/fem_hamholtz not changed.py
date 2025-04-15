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

        # --- Identify Solid Nodes for Fixed BCs ---
        solid_node_ids_all = torch.unique(self.solid_elements.flatten())
        solid_coords_all = self.nodes[solid_node_ids_all]
        solid_r_yz = torch.linalg.norm(solid_coords_all[:, 1:3], dim=1)
        
        outer_radius_tol = 1e-3
        end_plane_tol = 1e-3
        
        # Find solid nodes near x=0 and near r=r_outer
        fixed_solid_mask = (torch.abs(solid_coords_all[:, 0]) < end_plane_tol) & \
                           (torch.abs(solid_r_yz - r_outer) < outer_radius_tol)
                           
        self.fixed_solid_nodes_idx = solid_node_ids_all[fixed_solid_mask]
        print(f"[info] Identified {self.fixed_solid_nodes_idx.shape[0]} solid nodes at x=0 outer surface to fix.")
        if self.fixed_solid_nodes_idx.shape[0] < 3: # Need at least 3 non-collinear points generally
             print("[warning] Fewer than 3 solid nodes found to fix. Rigid body modes might not be fully constrained.")

        print(f"[info] 初始化完成, fluid elems: {self.fluid_elements.shape[0]}, solid elems: {self.solid_elements.shape[0]}, interface nodes: {self.interface_idx.shape[0]}, inlet nodes: {self.near_fluid_idx.shape[0]}, combined outlet nodes: {self.outlet_fluid_idx.shape[0]}, fixed solid nodes: {self.fixed_solid_nodes_idx.shape[0]}")
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
        plotter.add_mesh(fluid_mesh, color="cyan", opacity=0.5, label="Fluid Elements")
        plotter.add_mesh(solid_mesh, color="magenta", opacity=0.5, label="Solid Elements")
        plotter.add_mesh(interface_points, color="blue", point_size=10, render_points_as_spheres=True, label="Interface Nodes")
        plotter.add_mesh(inlet_points, color="yellow", point_size=10, render_points_as_spheres=True, label="Inlet Nodes")
        plotter.add_mesh(outlet_points, color="grey", point_size=10, render_points_as_spheres=True, label="Outlet Nodes")
        plotter.add_mesh(fixed_solid_points, color="purple", point_size=10, render_points_as_spheres=True, label="Fixed Solid Nodes")
        plotter.add_mesh(arrows, color="green", label="Interface Normals")
        plotter.add_legend()
        plotter.show()
    
    def assemble_global_system(self, E, nu, rho_s):
        """
        组装耦合系统 (Off-Diagonal Method):
          Fluid: A_f p = F_f
          Solid: A_s u = F_s
          Coupled: [[A_f, C_fs], [C_sf, A_s]] [p; u] = [F_f; F_s]
        """
        # ---- Fluid System Assembly (without coupling penalty) ----
        print("[info] Assembling fluid system...")
        A_f, F_f = self.assemble_global_fluid_system() # Ensure this func applies inlet/outlet BCs
        n_nodes = A_f.shape[0] # Total nodes, assuming fluid uses all

        # ---- Solid System Assembly (without coupling penalty) ----
        print("[info] Assembling solid system...")
        A_s, F_s, solid_mapping = self.assemble_global_solid_system(E, nu, rho_s)
        n_solid_dof = A_s.shape[0] # Total solid DOFs (n_solid_unique * 3)

        # ---- Assemble Coupling Matrices C_sf and C_fs ----
        print("[info] Assembling coupling matrices...")
        # Initialize coupling matrices with zeros
        # C_sf maps fluid pressure at node i to force on solid DOF j --> Size (n_solid_dof, n_nodes)
        C_sf = torch.zeros((n_solid_dof, n_nodes), dtype=torch.float32, device=device)
        # C_fs maps solid displacement at DOF j to fluid equation at node i --> Size (n_nodes, n_solid_dof)
        C_fs = torch.zeros((n_nodes, n_solid_dof), dtype=torch.float32, device=device)

        interface_node_set = set(self.interface_idx.cpu().numpy())

        # Iterate over FLUID elements to find interface FACES
        for elem_nodes in tqdm(self.fluid_elements, desc="Assembling coupling"):
            # Define the 4 faces (local indices)
            local_faces = [(0, 1, 2), (0, 3, 1), (0, 2, 3), (1, 3, 2)]

            for local_face in local_faces:
                global_node_indices_face = elem_nodes[torch.tensor(local_face, device=elem_nodes.device)] # Global node indices (3) for this face

                # Check if all 3 nodes are interface nodes
                is_interface_face = all(node_id.item() in interface_node_set for node_id in global_node_indices_face)

                if is_interface_face:
                    p0_idx, p1_idx, p2_idx = global_node_indices_face
                    p0, p1, p2 = self.nodes[p0_idx], self.nodes[p1_idx], self.nodes[p2_idx]

                    # Calculate face normal (pointing OUT of fluid element - needs check) and area
                    v1 = p1 - p0
                    v2 = p2 - p0
                    normal_vec = torch.cross(v1, v2) # Not normalized yet
                    face_area = torch.norm(normal_vec) / 2.0

                    if face_area > 1e-12:
                        normal_vec = normal_vec / (2.0 * face_area) # Normalized normal

                        # --- Orientation Check (Same as in normal calculation) ---
                        nodes_coords_elem = self.nodes[elem_nodes]
                        local_idx_p3 = list(set(range(4)) - set(local_face))[0]
                        p3 = nodes_coords_elem[local_idx_p3]
                        face_centroid = (p0 + p1 + p2) / 3.0
                        vec_to_p3 = p3 - face_centroid
                        if torch.dot(normal_vec, vec_to_p3) > 0:
                            normal_vec = -normal_vec # Ensure normal points outwards from fluid
                        # --- End Orientation Check ---

                        # Assemble contribution to C_sf (Pressure Force p*n on Solid)
                        # Integral over face: ∫ N_solid^T * n * N_fluid dΓ
                        # For linear elements, this integrates to (Area / 3) * n at each node
                        # So, fluid node `i` exerts force `(Area / 3) * n` on solid DOFs at node `i`
                        force_contrib = (face_area / 3.0) * normal_vec
                        for node_idx_tensor in global_node_indices_face:
                            node_idx = node_idx_tensor.item()
                            if node_idx in solid_mapping: # Check if this interface node is part of the solid system
                                solid_node_local_idx = solid_mapping[node_idx]
                                # Add force_contrib to rows corresponding to solid DOFs of this node
                                C_sf[solid_node_local_idx*3 : solid_node_local_idx*3+3, node_idx] += force_contrib

                        # Assemble contribution to C_fs (Effect of Solid Motion u on Fluid)
                        # Integral over face: ∫ N_fluid^T * (rho_f * omega^2 * n · u) dΓ = ∫ N_fluid^T * rho_f * omega^2 * n * N_solid dΓ * u
                        # Term linking solid DOF `j` to fluid node `i`
                        # Similar integral structure: (Area / 3) * rho_f * omega^2 * n
                        motion_contrib = rho_f * (self.omega**2) * (face_area / 3.0) * normal_vec
                        for node_idx_tensor in global_node_indices_face:
                             node_idx = node_idx_tensor.item()
                             if node_idx in solid_mapping:
                                 solid_node_local_idx = solid_mapping[node_idx]
                                 # Add motion_contrib to columns corresponding to solid DOFs
                                 C_fs[node_idx, solid_node_local_idx*3 : solid_node_local_idx*3+3] += motion_contrib


        # ---- Construct Global Block Matrix ----
        print("[info] Constructing global block matrix...")
        # Ensure correct dimensions before concatenation
        # A_f: (n_nodes, n_nodes)
        # A_s: (n_solid_dof, n_solid_dof)
        # C_fs: (n_nodes, n_solid_dof)
        # C_sf: (n_solid_dof, n_nodes)
        try:
             # Pad C_fs if n_solid_dof is smaller than expected (e.g., if solid_mapping is incomplete)
             if C_fs.shape[1] < n_solid_dof:
                  padding_fs = torch.zeros((n_nodes, n_solid_dof - C_fs.shape[1]), device=device)
                  C_fs = torch.cat((C_fs, padding_fs), dim=1)
             # Pad C_sf if n_nodes used in C_sf is smaller than n_nodes
             if C_sf.shape[1] < n_nodes:
                   padding_sf_cols = torch.zeros((n_solid_dof, n_nodes - C_sf.shape[1]), device=device)
                   C_sf = torch.cat((C_sf, padding_sf_cols), dim=1)
             # Pad A_s if n_solid_dof is smaller than expected
             if A_s.shape[0] < n_solid_dof or A_s.shape[1] < n_solid_dof:
                 padding_s = torch.zeros((n_solid_dof - A_s.shape[0], n_solid_dof), device=device)
                 A_s = torch.cat((A_s, padding_s), dim=0)
                 padding_s_cols = torch.zeros((n_solid_dof, n_solid_dof - A_s.shape[1]), device=device)
                 A_s = torch.cat((A_s, padding_s_cols), dim=1)


             top_row = torch.cat((A_f, C_fs), dim=1)
             # Pad C_sf rows if n_solid_dof used in C_sf is smaller than A_s rows
             if C_sf.shape[0] < n_solid_dof:
                  padding_sf_rows = torch.zeros((n_solid_dof - C_sf.shape[0], n_nodes), device=device)
                  C_sf = torch.cat((C_sf, padding_sf_rows), dim=0)

             bottom_row = torch.cat((C_sf, A_s), dim=1)
             A_global = torch.cat((top_row, bottom_row), dim=0)
        except RuntimeError as e:
             print(f"Error during block matrix construction: {e}")
             print(f"Shapes - A_f: {A_f.shape}, A_s: {A_s.shape}, C_fs: {C_fs.shape}, C_sf: {C_sf.shape}")
             print(f"n_nodes: {n_nodes}, n_solid_dof: {n_solid_dof}")
             raise

        # ---- Construct Global Force Vector ----
        F_global = torch.cat((F_f, F_s), dim=0)

        # ---- Apply Solid Fixed Boundary Conditions (AFTER full assembly) ----
        print("[info] Applying fixed solid boundary conditions...")
        dirichlet_penalty_solid = 1e10 # Use a large penalty value
        for node_idx in tqdm(self.fixed_solid_nodes_idx, desc="Applying fixed solid BCs"):
            # Map node index to solid DOF indices (remembering the n_nodes offset)
            dof_indices = [n_nodes + solid_mapping[node_idx.item()]*3 + i for i in range(3) if node_idx.item() in solid_mapping] # Use mapping

            for dof_idx in dof_indices:
                 if dof_idx < A_global.shape[0]: # Boundary check
                      A_global[dof_idx, :] = 0.0 # Zero out the row
                      A_global[:, dof_idx] = 0.0 # Zero out the column
                      A_global[dof_idx, dof_idx] = dirichlet_penalty_solid # Set diagonal to penalty
                      F_global[dof_idx] = 0.0 # Set RHS to 0
                 else:
                      print(f"[warning] Calculated solid DOF index {dof_idx} out of bounds for A_global shape {A_global.shape[0]}.")

        print("[info] Global matrix assembly complete.")
        return A_global, F_global, n_nodes, n_solid_dof # Return n_solid_dof

    def solve(self, E, nu, rho_s):
        """
        给定材料参数，组装全局系统并求解，返回：
          - 预测远端麦克风处流体声压（取 fluid 域 x≈1.0 点平均）
          - 全局解向量 u（其中 u[0:n_nodes] 为 fluid 声压）
        """
        A_global, F_global, n_nodes, n_solid_unique = self.assemble_global_system(E, nu, rho_s)
        print("[info] 开始求解")
        u = torch.linalg.solve(A_global + 1e-6 * torch.eye(A_global.shape[0], device=device), F_global)
        print("[info] 求解完成")
        # 从 fluid 部分取出远端麦克风处预测
        if self.mic_node_idx is not None:
             # Ensure mic_node_idx is within the fluid range
             if self.mic_node_idx < n_nodes:
                  p_mic = u[self.mic_node_idx]
             else:
                  print(f"[warning] Mic node index {self.mic_node_idx} is out of bounds for fluid DOFs ({n_nodes}). Setting p_mic to 0.")
                  p_mic = torch.tensor(0.0, device=device)
        else:
             print("[warning] Mic node index not found. Setting p_mic to 0.")
             p_mic = torch.tensor(0.0, device=device)
             
        return p_mic, u

    def assemble_global_fluid_system(self, c_f=343.0, source_value=1.0):
        """
        组装流体子系统矩阵 A_f 和向量 F_f (Helmholtz: K - k^2 M)。
        应用 Inlet (p=source_value) 和 Outlet (p=0) Dirichlet BCs。
        """
        n_nodes = self.nodes.shape[0]
        K_f = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)
        M_f = torch.zeros((n_nodes, n_nodes), dtype=torch.float32, device=device)
        F_f = torch.zeros(n_nodes, dtype=torch.float32, device=device)

        print("[debug] Assembling fluid K_f and M_f...")
        for elem in tqdm(self.fluid_elements, desc="Assembling fluid K/M"):
            coords = self.nodes[elem]
            K_e, M_e = element_matrices_fluid(coords)
            # Scatter K_e and M_e into global matrices
            for r in range(4):
                 for c in range(4):
                      K_f[elem[r], elem[c]] += K_e[r, c]
                      M_f[elem[r], elem[c]] += M_e[r, c]

        # Calculate A_f = K_f - k^2 * M_f
        k_sq = (self.omega / c_f)**2
        A_f = K_f - k_sq * M_f

        # Apply Dirichlet BCs using Penalty Method
        dirichlet_penalty_fluid = 1e10 # Penalty value

        # Inlet BC: p = source_value
        print(f"[debug] Applying fluid inlet BC (p={source_value}) to {self.near_fluid_idx.shape[0]} nodes...")
        for idx_tensor in self.near_fluid_idx:
             idx = idx_tensor.item()
             A_f[idx, :] = 0.0
             A_f[:, idx] = 0.0 # For symmetry
             A_f[idx, idx] = dirichlet_penalty_fluid
             F_f[idx] = dirichlet_penalty_fluid * source_value

        # Outlet BC: p = 0
        print(f"[debug] Applying fluid outlet BC (p=0) to {self.outlet_fluid_idx.shape[0]} nodes...")
        for idx_tensor in self.outlet_fluid_idx:
             idx = idx_tensor.item()
             # Avoid overwriting inlet BC if node is somehow both
             if A_f[idx, idx] != dirichlet_penalty_fluid:
                  A_f[idx, :] = 0.0
                  A_f[:, idx] = 0.0 # For symmetry
                  A_f[idx, idx] = dirichlet_penalty_fluid
                  F_f[idx] = 0.0 # Penalty * 0

        print("[debug] Fluid system assembly complete.")
        return A_f, F_f

    def assemble_global_solid_system(self, E, nu, rho_s):
        """
        组装固体子系统矩阵 A_s 和向量 F_s (K - omega^2 M)。
        Handles mapping from global node indices to local solid DOFs.
        Returns: A_s, F_s, solid_mapping
        """
        solid_unique_nodes = torch.unique(self.solid_elements.flatten())
        solid_mapping = {node_id.item(): i for i, node_id in enumerate(solid_unique_nodes)}
        n_solid_nodes_unique = len(solid_unique_nodes)
        n_solid_dof = n_solid_nodes_unique * 3

        K_s = torch.zeros((n_solid_dof, n_solid_dof), dtype=torch.float32, device=device)
        M_s = torch.zeros((n_solid_dof, n_solid_dof), dtype=torch.float32, device=device)
        F_s = torch.zeros(n_solid_dof, dtype=torch.float32, device=device)

        print("[debug] Assembling solid K_s and M_s...")
        for elem in tqdm(self.solid_elements, desc="Assembling solid K/M"):
            coords = self.nodes[elem]
            K_e, M_e = element_matrices_solid(coords, E, nu, rho_s) # K_e, M_e are 12x12

            # Scatter K_e and M_e into global solid matrices using mapping
            for r_local in range(4): # Local node index within element (0-3)
                 global_node_r = elem[r_local].item()
                 if global_node_r not in solid_mapping: continue # Should not happen if solid_unique_nodes is correct
                 solid_idx_r = solid_mapping[global_node_r] # Index within the solid-only system

                 for c_local in range(4):
                      global_node_c = elem[c_local].item()
                      if global_node_c not in solid_mapping: continue
                      solid_idx_c = solid_mapping[global_node_c]

                      # Get the 3x3 block from K_e and M_e
                      K_block = K_e[r_local*3 : (r_local+1)*3, c_local*3 : (c_local+1)*3]
                      M_block = M_e[r_local*3 : (r_local+1)*3, c_local*3 : (c_local+1)*3]

                      # Add to the global solid matrices at the mapped indices
                      K_s[solid_idx_r*3 : (solid_idx_r+1)*3, solid_idx_c*3 : (solid_idx_c+1)*3] += K_block
                      M_s[solid_idx_r*3 : (solid_idx_r+1)*3, solid_idx_c*3 : (solid_idx_c+1)*3] += M_block

        # Calculate A_s = K_s - omega^2 * M_s
        A_s = K_s - (self.omega**2) * M_s

        print("[debug] Solid system assembly complete.")
        return A_s, F_s, solid_mapping
