# jax_fem.py
import jax
import jax.numpy as jnp
import meshio
import numpy as np
from typing import Tuple
ENTER_FUNC_PRINT = True
EXIT_FUNC_PRINT = True
R_inner_main = 0.045  # 主管内半径，单位：米

# -------------------------
# 1. 网格读取与处理
# -------------------------
def load_mesh(msh_file: str):
    """
    使用 meshio 读取 gmsh 网格文件，假设：
      - 流体域体单元（四面体）物理标签为 1；
      - 固体域体单元（四面体）物理标签为 2。
    返回：
      nodes: jnp.array shape (n_nodes,3)
      elements_fluid: jnp.array shape (n_elem_fluid,4)   ——节点索引（0-based）
      elements_solid: jnp.array shape (n_elem_solid,4)
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering load_mesh function...")
    mesh = meshio.read(msh_file)
    nodes = mesh.points[:, :3]
    # 提取四面体单元
    elements_fluid = np.array(mesh.cells[0].data, dtype=np.int32)
    elements_solid = np.array(mesh.cells[1].data, dtype=np.int32)
    
    # 预处理界面节点索引
    tol = 1e-3 # 1mm
    r = np.sqrt(nodes[:,1]**2 + nodes[:,2]**2)
    interface_mask = np.abs(r - R_inner_main) < tol
    interface_indices = np.nonzero(interface_mask)[0].astype(np.int32)
    print(f"elements_fluid shape: {elements_fluid.shape}, elements_solid shape: {elements_solid.shape}, interface_indices shape: {interface_indices.shape}")
    if EXIT_FUNC_PRINT:
        print(f"Exiting load_mesh function...")
    return jnp.array(nodes), jnp.array(elements_fluid), jnp.array(elements_solid), jnp.array(interface_indices, dtype=jnp.int32)

# -------------------------
# 2. 四面体单元基础计算
# -------------------------
def tetra_volume(nodes_elem: jnp.ndarray) -> jnp.ndarray:
    """
    计算四面体单元体积
    Volume = 1/6 * abs(det([p1-p0, p2-p0, p3-p0]))
    nodes_elem: jnp.array shape (4,3)
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering tetra_volume function...")
    p0 = nodes_elem[0]
    v1 = nodes_elem[1] - p0
    v2 = nodes_elem[2] - p0
    v3 = nodes_elem[3] - p0
    det_val = jnp.linalg.det(jnp.stack([v1, v2, v3], axis=1))
    if EXIT_FUNC_PRINT:
        print(f"Exiting tetra_volume function...")
    return jnp.abs(det_val) / 6.0

# -------------------------
# 3. 单元矩阵组装——流体（Helmholtz）
# -------------------------
def element_matrices_fluid(nodes_elem: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    对于流体域：求解 Helmholtz 方程：∇²p + k² p = 0，其中 k=omega/c_f.
    用线性 tetrahedral 元件求刚度矩阵和一致质量矩阵。
    使用解析公式：梯度的形函数常数，刚度矩阵K_e = V * (B^T B), 质量矩阵M_e = (V/10)*(1+δ_ij)。
    返回单元矩阵 A_e = K_e - k² M_e, shape (4,4)
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering element_matrices_fluid function...")
    V = tetra_volume(nodes_elem)
    # 构造增广矩阵 A = [1, x, y, z] (4x4)
    ones = jnp.ones((4, 1))
    A = jnp.concatenate([ones, nodes_elem], axis=1)  # shape (4,4)
    invA = jnp.linalg.inv(A)
    # 每个形函数 N_i = a_i + b_i*x + c_i*y + d_i*z, 梯度 = (b_i, c_i, d_i)
    grads = invA[1:4, :]  # shape (3,4)；第 i 列为 grad(N_i)
    K_e = jnp.dot(grads.T, grads) * V  # 刚度矩阵
    M_e = (V / 10.0) * (jnp.ones((4,4)) + jnp.eye(4))  # 一致质量矩阵
    # 波数 k
    c = 343.0  # 声速, 单位 m/s（流体参数已知）
    k = omega / c
    A_e = K_e - (k**2) * M_e
    if EXIT_FUNC_PRINT:
        print(f"Exiting element_matrices_fluid function...")
    return A_e

# -------------------------
# 4. 单元矩阵组装——固体（线弹性）
# -------------------------
def element_matrices_solid(nodes_elem: jnp.ndarray, E: float, nu: float, rho_s: float, omega: float) -> jnp.ndarray:
    """
    对于固体域：求解线弹性动问题 (谐振分析)：
      方程：K_e u - ω² M_e u = 0
    采用 3D 线弹性四面体单元（每节点 3 d.o.f.）
    先计算形函数梯度（常量）并构造 B 矩阵，B 的尺寸为 (6, 12)。
    标准线性弹性材料的 D 矩阵 (6,6)：对于各向同性材料，
      D = E/((1+ν)(1-2ν))*[[1-ν, ν, ν, 0, 0, 0],
                           [ν, 1-ν, ν, 0, 0, 0],
                           [ν, ν, 1-ν, 0, 0, 0],
                           [0, 0, 0, (1-2ν)/2, 0, 0],
                           [0, 0, 0, 0, (1-2ν)/2, 0],
                           [0, 0, 0, 0, 0, (1-2ν)/2]]
    单元刚度： K_e = B^T D B * V, 单元一致质量矩阵 M_e 取自密度 ρₛ * V/4 分布于各节点。
    返回单元矩阵 A_e = K_e - ω² M_e, shape (12,12)
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering element_matrices_solid function...")
    V = tetra_volume(nodes_elem)
    ones = jnp.ones((4,1))
    A_mat = jnp.concatenate([ones, nodes_elem], axis=1)  # (4,4)
    invA = jnp.linalg.inv(A_mat)
    grads = invA[1:4, :]  # shape (3,4)；各列 i 为 grad(N_i)
    # 构造 B 矩阵, 6 x 12
    def B_i(i):
        dN = grads[:, i]
        return jnp.array([
            [dN[0],      0,      0],
            [     0, dN[1],      0],
            [     0,      0, dN[2]],
            [dN[1], dN[0],      0],
            [     0, dN[2], dN[1]],
            [dN[2],      0, dN[0]]
        ])
    B_e = jnp.hstack([B_i(i) for i in range(4)])  # shape (6, 12)
    # 弹性矩阵 D
    factor = E / ((1+nu)*(1-2*nu))
    D = factor * jnp.array([
        [1-nu,   nu,    nu,       0,       0,       0],
        [  nu, 1-nu,    nu,       0,       0,       0],
        [  nu,   nu,  1-nu,       0,       0,       0],
        [   0,    0,    0, (1-2*nu)/2,  0,       0],
        [   0,    0,    0,       0, (1-2*nu)/2,  0],
        [   0,    0,    0,       0,       0, (1-2*nu)/2]
    ])
    K_e = jnp.dot(B_e.T, jnp.dot(D, B_e)) * V
    # 一致质量矩阵：采用 lumped mass
    m = rho_s * V / 4.0
    M_e = m * jnp.eye(12)
    A_e = K_e - (omega**2)*M_e
    if EXIT_FUNC_PRINT:
        print(f"Exiting element_matrices_solid function...")
    return A_e

# -------------------------
# 5. 全局矩阵组装
# -------------------------
def assemble_global_system(nodes: jnp.ndarray, elements: jnp.ndarray, element_func, omega: float, E=None, nu=None, rho_s=None, dof_per_node: int = 1):
    """
    组装全局矩阵。  
    nodes: jnp.array (n_nodes, 3)
    elements: jnp.array (n_elements, 4) 四面体单元，节点索引
    element_func: 对单元调用函数。对于流体，传入 nodes_elem 和 omega；对于固体，传入 (nodes_elem, E, nu, rho_s, omega)
    dof_per_node: 每节点自由度, 流体:1, 固体:3.
    返回全局矩阵 A (dense) 和右侧向量 f (本例中无体载荷 f=0)。
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering assemble_global_system function...")
    n_nodes = nodes.shape[0]
    n_dof = n_nodes * dof_per_node
    A_global = jnp.zeros((n_dof, n_dof))
    f_global = jnp.zeros((n_dof,))
    # 对每个单元循环组装（工业级可采用向量化或并行汇总，此处使用 fori_loop）
    def body_fun(i, A_acc):
        elem = elements[i]  # 4 个节点
        nodes_elem = nodes[elem, :]  # shape (4,3)
        if E is None:
            A_e = element_func(nodes_elem, omega)  # fluid: shape (4,4)
            local_dof = 4  # 4 nodes * 1
        else:
            A_e = element_func(nodes_elem, E, nu, rho_s, omega)  # solid: shape (12,12)
            local_dof = 4 * dof_per_node  # 4 * 3
        # 全局索引
        indices = []
        for n in elem:
            for a in range(dof_per_node):
                indices.append((n.astype(int))*dof_per_node + a)
        indices = jnp.array(indices)
        A_acc = A_acc.at[jnp.ix_(indices, indices)].add(A_e)
        return A_acc
    A_global = jax.lax.fori_loop(0, elements.shape[0], body_fun, A_global)
    if EXIT_FUNC_PRINT:
        print(f"Exiting assemble_global_system function...")
    return A_global, f_global

# -------------------------
# 6. 流固耦合矩阵组装（Penalty 方法）
# -------------------------
def assemble_coupling_system(nodes: jnp.ndarray, fluid_elements: jnp.ndarray, solid_elements: jnp.ndarray,
                               omega: float, E: float, nu: float, rho_s: float, interface_indices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    组装流体与固体的耦合系统。  
    假定 fluid 元素每节点 1 d.o.f., solid 元素每节点 3 d.o.f., 且两者基于相同的节点集合（网格中所有节点均出现在两个物理域中）。
    采用 penalty 方法在流固界面上施加：
         r = p - (u · n) = 0,
    其中 n 为外侧单位法向量（通过节点坐标计算，从几何条件确定：对于主管，假设 n = (0, y/√(y²+z²), z/√(y²+z²))）。
    对于每个界面节点（满足 |sqrt(y²+z²) - R_inner_main| < tol），在流体方程中加 α, 在固体方程中加入 -α * n_y, -α * n_z 等。
    全局自由度排序：先 fluid (n_nodes * 1), 后固体 (n_nodes * 3).
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering assemble_coupling_system function...")
    nodes_fluid = nodes  # 所有节点
    nodes_solid = nodes  # 假设相同
    dof_fluid = nodes_fluid.shape[0]  # 1 d.o.f. 每节点
    dof_solid = nodes_solid.shape[0] * 3  # 3 d.o.f.
    # 组装 fluid 和 solid 全局矩阵
    A_fluid, f_fluid = assemble_global_system(nodes, fluid_elements, element_matrices_fluid, omega, dof_per_node=1)
    A_solid, f_solid = assemble_global_system(nodes, solid_elements, element_matrices_solid, omega, E, nu, rho_s, dof_per_node=3)
    # 全局系统：按流体与固体并排
    A_global = jnp.block([
        [A_fluid, jnp.zeros((A_fluid.shape[0], A_solid.shape[1]))],
        [jnp.zeros((A_solid.shape[0], A_fluid.shape[1])), A_solid]
    ])
    f_global = jnp.concatenate([f_fluid, f_solid], axis=0)
    # 边界耦合：确定流固界面节点(interface_indices， 已经预处理)
    
    # 对每个界面节点，计算外侧法向量 n = (0, y/√(y²+z²), z/√(y²+z²))
    normals = []
    for i in interface_indices:
        coord = nodes[i]
        r = jnp.sqrt(coord[1]**2 + coord[2]**2) + 1e-8
        n_vec = jnp.array([0.0, coord[1]/r, coord[2]/r])
        normals.append(n_vec)
    normals = jnp.stack(normals, axis=0)  # shape (n_interface, 3)
    # 对每个界面节点, 在全局系统中对应 fluid dof index = i, solid dof indices = (i*3, i*3+1, i*3+2) but offset by dof_fluid.
    penalty = 1e8
    for idx, n_vec in zip(interface_indices, normals):
        i_f = idx  # fluid dof index (scalar)
        # solid: global index offset = dof_fluid + idx*3
        i_s0 = idx*3 + dof_fluid
        # 组装对流固界面 penalty：
        # 式子： penalty*(p - (u·n))^2  对 p, u 施加耦合。求二阶导后加到全局矩阵中。
        A_global = A_global.at[i_f, i_f].add(penalty)
        A_global = A_global.at[i_f, i_s0+1].add(-penalty * n_vec[1])
        A_global = A_global.at[i_f, i_s0+2].add(-penalty * n_vec[2])
        A_global = A_global.at[i_s0+1, i_f].add(-penalty * n_vec[1])
        A_global = A_global.at[i_s0+2, i_f].add(-penalty * n_vec[2])
        A_global = A_global.at[i_s0+1, i_s0+1].add(penalty * n_vec[1]**2)
        A_global = A_global.at[i_s0+2, i_s0+2].add(penalty * n_vec[2]**2)
    if EXIT_FUNC_PRINT:
        print(f"Exiting assemble_coupling_system function...")
    return A_global, f_global

# -------------------------
# 7. 全局系统求解
# -------------------------
def solve_fsi_system(E: float, nu: float, rho_s: float, omega: float, mesh_data):
    """
    组装并求解流固耦合系统，返回全局解向量 u (dense)。
    mesh_data: (nodes, elements_fluid, elements_solid, interface_indices)
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering solve_fsi_system function...")
    nodes, elem_fluid, elem_solid, interface_indices = mesh_data
    A_global, f_global = assemble_coupling_system(nodes, elem_fluid, elem_solid, omega, E, nu, rho_s, interface_indices)
    u = jnp.linalg.solve(A_global, f_global)
    if EXIT_FUNC_PRINT:
        print(f"Exiting solve_fsi_system function...")
    return u

# -------------------------
# 8. 利用求解结果在麦克风处做插值
# -------------------------
def interpolate_pressure(u: jnp.ndarray, nodes: jnp.ndarray, mic_pos: jnp.ndarray) -> jnp.ndarray:
    """
    使用 JAX 实现最近邻插值，避免 NumPy 操作。
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering interpolate_pressure function...")
    n_nodes = nodes.shape[0]
    p_fluid = u[:n_nodes]
    # 计算所有节点到麦克风位置的距离（使用 JAX 操作）
    distances = jnp.linalg.norm(nodes - mic_pos, axis=1)
    idx = jnp.argmin(distances)
    if EXIT_FUNC_PRINT:
        print(f"Exiting interpolate_pressure function...")
    return p_fluid[idx]

# -------------------------
# 9. 对外接口：正向仿真函数
# -------------------------
@jax.jit
def forward_fsi(E: float, nu: float, rho_s: float, omega: float, mesh_data, mic_pos: jnp.ndarray) -> jnp.ndarray:
    """
    正向仿真接口：给定材料参数 (E, nu, rho_s) 及激励频率 omega，
    组装求解流固耦合系统，并返回远端麦克风处的预测声压 (标量)
    """
    if ENTER_FUNC_PRINT:
        print(f"Entering forward_fsi function...")
    u = solve_fsi_system(E, nu, rho_s, omega, mesh_data)
    nodes, _, _, _ = mesh_data
    p_mic = interpolate_pressure(u, nodes, mic_pos)
    if EXIT_FUNC_PRINT:
        print(f"Exiting forward_fsi function...")
    return p_mic


