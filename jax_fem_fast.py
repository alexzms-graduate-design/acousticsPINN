# jax_fem_fast.py
import jax
import jax.numpy as jnp
import meshio
import numpy as np
from typing import Tuple
from functools import partial
from jax import debug

from jax.config import config
config.update("jax_debug_nans", True)
# float64
config.update("jax_enable_x64", True)

# 禁用调试打印以提升性能
ENTER_FUNC_PRINT = False
EXIT_FUNC_PRINT = False
print("Import jax_fem_fast.py successfully!")
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
    
    volumes = [tetra_volume(nodes[elem]) for elem in mesh.cells[0].data]
    if any(v <= 0 for v in volumes):
        raise ValueError("存在无效单元体积（≤0），请调整网格参数！")
    
    elements_fluid = np.array(mesh.cells[0].data, dtype=np.int32)
    elements_solid = np.array(mesh.cells[1].data, dtype=np.int32)
    
    # 预处理界面节点索引
    tol = 1e-3
    r = np.sqrt(nodes[:,1]**2 + nodes[:,2]**2)
    interface_mask = np.abs(r - R_inner_main) < tol
    interface_indices = np.nonzero(interface_mask)[0].astype(np.int32)
    
    # 静态形状断言确保JAX追踪稳定性
    assert elements_fluid.shape[1] == 4, "elements_fluid 必须为 (n,4)"
    assert elements_solid.shape[1] == 4, "elements_solid 必须为 (n,4)"
    print(f"elements_fluid.shape: {elements_fluid.shape}, elements_solid.shape: {elements_solid.shape}, interface_indices.shape: {interface_indices.shape}")
    if EXIT_FUNC_PRINT:
        print(f"Exiting load_mesh function...")
    return (jnp.array(nodes), 
            jnp.array(elements_fluid), 
            jnp.array(elements_solid), 
            jnp.array(interface_indices, dtype=jnp.int32))

# -------------------------
# 2. 四面体单元基础计算（JIT预编译）
# -------------------------
@jax.jit
def tetra_volume(nodes_elem: jnp.ndarray) -> jnp.ndarray:
    """
    计算四面体单元体积
    Volume = 1/6 * abs(det([p1-p0, p2-p0, p3-p0]))
    nodes_elem: jnp.array shape (4,3)
    """
    p0 = nodes_elem[0]
    v1 = nodes_elem[1] - p0
    v2 = nodes_elem[2] - p0
    v3 = nodes_elem[3] - p0
    det_val = jnp.linalg.det(jnp.stack([v1, v2, v3], axis=1))
    det_val = jnp.where(jnp.abs(det_val) < 1e-12, 1e-12, det_val)
    return jnp.abs(det_val) / 6.0

# -------------------------
# 3. 单元矩阵组装——流体（Helmholtz，向量化）
# -------------------------
@partial(jax.jit, static_argnames=('omega',))
def element_matrices_fluid(nodes_elem: jnp.ndarray, omega: float) -> jnp.ndarray:
    """
    对于流体域：求解 Helmholtz 方程：∇²p + k² p = 0，其中 k=omega/c_f.
    用线性 tetrahedral 元件求刚度矩阵和一致质量矩阵。
    返回单元矩阵 A_e = K_e - k² M_e, shape (4,4)
    """
    V = tetra_volume(nodes_elem)
    ones = jnp.ones((4, 1))
    A = jnp.concatenate([ones, nodes_elem], axis=1)
    A_reg = A + 1e-6 * jnp.eye(A.shape[0])  # 防止奇异矩阵
    invA = jnp.linalg.inv(A_reg)
    grads = invA[1:4, :]
    K_e = jnp.dot(grads.T, grads) * V
    M_e = (V / 10.0) * (jnp.ones((4,4)) + jnp.eye(4))
    c = 343.0  # 声速固定参数
    k = omega / c
    return K_e - (k**2) * M_e

# -------------------------
# 4. 单元矩阵组装——固体（线弹性，向量化）
# -------------------------
@partial(jax.jit, static_argnames=('E', 'nu', 'rho_s', 'omega'))
def element_matrices_solid(nodes_elem: jnp.ndarray, E: float, nu: float, 
                          rho_s: float, omega: float) -> jnp.ndarray:
    """
    对于固体域：求解线弹性动问题 (谐振分析)
    返回单元矩阵 A_e = K_e - ω² M_e, shape (12,12)
    """
    V = tetra_volume(nodes_elem)
    ones = jnp.ones((4,1))
    A_mat = jnp.concatenate([ones, nodes_elem], axis=1)
    invA = jnp.linalg.inv(A_mat)
    grads = invA[1:4, :]
    
    # 向量化构造B矩阵
    def B_i(i):
        dN = grads[:, i]
        return jnp.array([
            [dN[0], 0, 0],
            [0, dN[1], 0],
            [0, 0, dN[2]],
            [dN[1], dN[0], 0],
            [0, dN[2], dN[1]],
            [dN[2], 0, dN[0]]
        ])
    B_e = jnp.hstack([B_i(i) for i in range(4)])
    
    # 弹性矩阵D
    factor = E / ((1+nu)*(1-2*nu))
    D = factor * jnp.array([
        [1-nu, nu, nu, 0, 0, 0],
        [nu, 1-nu, nu, 0, 0, 0],
        [nu, nu, 1-nu, 0, 0, 0],
        [0, 0, 0, (1-2*nu)/2, 0, 0],
        [0, 0, 0, 0, (1-2*nu)/2, 0],
        [0, 0, 0, 0, 0, (1-2*nu)/2]
    ])
    K_e = jnp.dot(B_e.T, jnp.dot(D, B_e)) * V
    m = rho_s * V / 4.0
    M_e = m * jnp.eye(12)
    return K_e - (omega**2)*M_e

# -------------------------
# 5. 全局矩阵组装（向量化优化）
# -------------------------
@partial(jax.jit, static_argnames=('element_func', 'dof_per_node'))
def assemble_global_system(nodes: jnp.ndarray, elements: jnp.ndarray, element_func, 
                          omega: float, E=None, nu=None, rho_s=None, dof_per_node: int = 1):
    """
    向量化全局矩阵组装：使用vmap批处理单元计算
    """
    n_nodes = nodes.shape[0]
    n_dof = n_nodes * dof_per_node
    A_global = jnp.zeros((n_dof, n_dof))
    
    # 批处理计算所有单元矩阵
    def compute_element_matrix(elem):
        nodes_elem = nodes[elem]
        if E is None:
            return element_func(nodes_elem, omega)
        else:
            return element_func(nodes_elem, E, nu, rho_s, omega)
    all_A_e = jax.vmap(compute_element_matrix)(elements)
    
    # 向量化累加
    def scatter(A_acc, elem_and_A_e):
        elem, A_e = elem_and_A_e
        indices = (elem[:, None] * dof_per_node + jnp.arange(dof_per_node)).flatten()
        return A_acc.at[jnp.ix_(indices, indices)].add(A_e), None
    
    A_global, _ = jax.lax.scan(scatter, A_global, (elements, all_A_e))
    return A_global, jnp.zeros(n_dof)

# -------------------------
# 6. 流固耦合矩阵组装（向量化惩罚项）
# -------------------------
@partial(jax.jit, static_argnames=('E', 'nu', 'rho_s'))
def assemble_coupling_system(nodes: jnp.ndarray, fluid_elements: jnp.ndarray, solid_elements: jnp.ndarray,
                             omega: float, E: float, nu: float, rho_s: float, 
                             interface_indices: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    向量化处理流固耦合惩罚项
    """
    # 组装基础矩阵
    A_fluid, _ = assemble_global_system(nodes, fluid_elements, element_matrices_fluid, omega, dof_per_node=1)
    A_solid, _ = assemble_global_system(nodes, solid_elements, element_matrices_solid, omega, E, nu, rho_s, dof_per_node=3)
    A_global = jnp.block([[A_fluid, jnp.zeros((A_fluid.shape[0], A_solid.shape[1]))],
                          [jnp.zeros((A_solid.shape[0], A_fluid.shape[1])), A_solid]])
    
    # 向量化法向量计算
    def compute_normal(i):
        coord = nodes[i]
        r = jnp.sqrt(coord[1]**2 + coord[2]**2) + 1e-8
        return jnp.array([0.0, coord[1]/r, coord[2]/r])
    normals = jax.vmap(compute_normal)(interface_indices)
    
    # 向量化添加惩罚项
    penalty = 1e8
    def add_penalty(A, idx_nvec):
        idx, n_vec = idx_nvec
        i_f = idx
        i_s0 = idx*3 + nodes.shape[0]  # 流体自由度总数
        updates = [
            (i_f, i_f, penalty),
            (i_f, i_s0+1, -penalty * n_vec[1]),
            (i_f, i_s0+2, -penalty * n_vec[2]),
            (i_s0+1, i_f, -penalty * n_vec[1]),
            (i_s0+2, i_f, -penalty * n_vec[2]),
            (i_s0+1, i_s0+1, penalty * n_vec[1]**2),
            (i_s0+2, i_s0+2, penalty * n_vec[2]**2)
        ]
        for row, col, val in updates:
            A = A.at[row, col].add(val)
        return A
    return jax.lax.fori_loop(0, len(interface_indices), 
                             lambda i, A: add_penalty(A, (interface_indices[i], normals[i])),
                             A_global), jnp.zeros(A_global.shape[0])

# -------------------------
# 7. 全局系统求解（保持原始接口）
# -------------------------
@jax.jit
def solve_fsi_system(E: float, nu: float, rho_s: float, omega: float, mesh_data):
    nodes, elem_fluid, elem_solid, interface_indices = mesh_data
    A_global, f_global = assemble_coupling_system(nodes, elem_fluid, elem_solid, omega, E, nu, rho_s, interface_indices)
    
    # 添加正则化项防止奇异矩阵
    A_regularized = A_global + 1e-6 * jnp.eye(A_global.shape[0])
    u = jnp.linalg.solve(A_regularized, f_global)
    debug.callback(lambda x: print(f"Max displacement: {jnp.max(jnp.abs(x))}"), u)
    return u

# -------------------------
# 8. 麦克风插值（纯JAX实现）
# -------------------------
@jax.jit
def interpolate_pressure(u: jnp.ndarray, nodes: jnp.ndarray, mic_pos: jnp.ndarray) -> jnp.ndarray:
    n_nodes = nodes.shape[0]
    p_fluid = u[:n_nodes]
    distances = jnp.linalg.norm(nodes - mic_pos, axis=1)
    return p_fluid[jnp.argmin(distances)]

# -------------------------
# 9. 对外接口（JIT顶层封装）
# -------------------------
@jax.jit
def forward_fsi(E: float, nu: float, rho_s: float, omega: float, mesh_data, mic_pos: jnp.ndarray) -> jnp.ndarray:
    u = solve_fsi_system(E, nu, rho_s, omega, mesh_data)
    if EXIT_FUNC_PRINT:
        print(f"Exiting forward_fsi function...")
    return interpolate_pressure(u, mesh_data[0], mic_pos)