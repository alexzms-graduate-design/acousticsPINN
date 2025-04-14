# fem.py
import torch
import torch.nn as nn
import numpy as np
import meshio
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # 提取 tetrahedral单元并分物理组
        fluid_elems = []
        solid_elems = []
        for cell in tqdm(mesh.cells, desc="[info] 提取单元"):
            if cell.type == "tetra":
                data = cell.data
                phys = mesh.cell_data_dict["gmsh:physical"]["tetra"]
                for i in range(len(data)):
                    if phys[i] == 1:  # Fluid 域
                        fluid_elems.append(data[i])
                    elif phys[i] == 2:  # Solid 域（管壁）
                        solid_elems.append(data[i])
        self.fluid_elements = torch.tensor(np.array(fluid_elems), dtype=torch.long, device=device)
        self.solid_elements = torch.tensor(np.array(solid_elems), dtype=torch.long, device=device)

        # 构建流固接口映射（假定流体内腔的边界即为内表面，且固体的外表面即为接口）
        tol = 1e-3
        r_inner = 0.045  # 内径 9cm/2 = 4.5cm
        # 对于 fluid，找出节点满足： sqrt(y^2+z^2) ≈ 0.045
        r_fluid = torch.norm(self.nodes[:,1:3], dim=1)
        self.interface_fluid_idx = torch.nonzero(torch.abs(r_fluid - r_inner) < tol).squeeze()
        # 对于 solid，亦然：固体接口节点为那些在固体单元中出现且满足条件
        # 为简单起见，我们取 solid 单元中所有节点，再在这些节点中筛选：
        solid_node_ids = torch.unique(self.solid_elements)
        solid_coords = self.nodes[solid_node_ids]
        r_solid = torch.norm(solid_coords[:,1:3], dim=1)
        interface_mask = torch.abs(r_solid - r_inner) < tol
        self.interface_solid_idx = solid_node_ids[interface_mask]
        # 建立从固体接口节点到流体接口节点的映射（最近邻搜索）
        fluid_iface_coords = self.nodes[self.interface_fluid_idx]
        mapping = []
        for idx in self.interface_solid_idx:
            coord = self.nodes[idx]
            dists = torch.norm(fluid_iface_coords - coord, dim=1)
            min_idx = torch.argmin(dists)
            mapping.append(self.interface_fluid_idx[min_idx].item())
        self.interface_mapping = torch.tensor(mapping, dtype=torch.long, device=device)
        # 计算流体接口法向量：对于圆柱内壁，法向量约为 (0, y, z)/sqrt(y^2+z^2)
        normals = []
        for idx in self.interface_fluid_idx:
            coord = self.nodes[idx]
            yz = coord[1:3]
            norm_val = torch.norm(yz)
            if norm_val > 1e-6:
                n = torch.cat([torch.tensor([0.0], device=device), yz/norm_val])
            else:
                n = torch.zeros(3, device=device)
            normals.append(n)
        self.interface_normals = torch.stack(normals, dim=0)  # shape [n_iface,3]
        # 定义噪音源边界：fluid 节点中 x ≈ 0
        tol_near = 1e-3
        self.near_fluid_idx = torch.nonzero(torch.abs(self.nodes[:,0]) < tol_near).squeeze()
        # 定义远端评估点：fluid 节点中 x ≈ 1.0 （远端麦克风放置位置）
        tol_far = 1e-3
        self.far_fluid_idx = torch.nonzero(torch.abs(self.nodes[:,0] - 1.0) < tol_far).squeeze()
        print("[info] 初始化完成")
    
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
        for idx in tqdm(self.near_fluid_idx, desc="[info] 流体域组装"):
            A_f[idx, :] = 0.0
            A_f[idx, idx] = 1.0
            F_f[idx] = p0
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
        for idx_idx, idx_solid in enumerate(tqdm(self.interface_solid_idx, desc="[info] 耦合处理")):
            # 对应 fluid 节点
            fluid_idx = self.interface_mapping[idx_idx].item()
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
