# main.py
import torch
import numpy as np
import jax
import jax.numpy as jnp
import time
from jax_fem_fast import load_mesh, forward_fsi

# -------------------------
# 1. 基本参数与网格数据加载
# -------------------------
MESH_FILE = "y_pipe.msh"   # gmsh 生成的网格文件
nodes, elem_fluid, elem_solid, interface_indices = load_mesh(MESH_FILE)
mesh_data = (nodes, elem_fluid, elem_solid, interface_indices)

# 激励频率 (rad/s): 例如，对于 1000Hz, omega = 2π*1000
freq = 100.0
omega = 2 * np.pi * freq

# 定义入口激励：在流体域（dof_per_node==1）的入口节点 (x≈0) 固定 p=1.0 Pa
nodes = np.array(mesh_data[0])  # 将 jax 数组转换为 NumPy 以便筛选
tol_inlet = 1e-3
source_mask = np.abs(nodes[:,0]) < tol_inlet
source_indices = np.nonzero(source_mask)[0].astype(np.int32)
source_indices = jnp.array(source_indices, dtype=jnp.int32)  # 转为 jax 数组
print(f"source_indices: {source_indices}")  
source_value = 1.0  # 单位 Pa

# 远端麦克风位置 (远端位于主管中心点 x=1.0, y=0, z=0)
mic_pos = jnp.array([1.0, 0.0, 0.0])

# 远端测量声压（实测值），例如 0.5 Pa（工业中由传感器采集）
meas_p = 0.5

# -------------------------
# 2. 定义 PyTorch 自定义算子，调用 JAX FEM 正向仿真
# -------------------------
class JaxFEMMicPressure(torch.autograd.Function):
    """
    自定义 PyTorch 算子，利用 JAX 进行 3D 流固耦合 FEM 求解，
    输出远端麦克风处的声压预测值。
    """
    @staticmethod
    def forward(ctx, E_t, nu_t, rho_t):
        # 将 torch 参数转换为 numpy/jax 数组（标量）
        E_np = E_t.detach().cpu().numpy()
        nu_np = nu_t.detach().cpu().numpy()
        rho_np = rho_t.detach().cpu().numpy()
        # 调用 JAX FEM 正向函数 (已 JIT 编译)
        p_pred = forward_fsi(float(E_np), float(nu_np), float(rho_np), omega, mesh_data, mic_pos, source_indices, source_value)
        print(f"p_pred: {p_pred}")
        # 获取梯度信息：使用 jax.value_and_grad
        def fem_obj(E_in, nu_in, rho_in):
            return forward_fsi(E_in, nu_in, rho_in, omega, mesh_data, mic_pos, source_indices, source_value)
        # 求值和梯度（返回标量和对应梯度元组）
        print("Entering jax.value_and_grad...")
        p_val, grads = jax.value_and_grad(fem_obj, argnums=(0,1,2))(float(E_np), float(nu_np), float(rho_np))
        print(f"p_val: {p_val}, grads[0]: {grads[0]}, grads[1]: {grads[1]}, grads[2]: {grads[2]}")
        print("Exiting jax.value_and_grad...")
        grad_E, grad_nu, grad_rho = grads
        # 保存梯度至 ctx（转换为 torch.tensor）
        print("Entering ctx.save_for_backward...")
        ctx.save_for_backward(torch.tensor([float(grad_E)], dtype=torch.float32),
                              torch.tensor([float(grad_nu)], dtype=torch.float32),
                              torch.tensor([float(grad_rho)], dtype=torch.float32))
        print("Exiting ctx.save_for_backward...")
        return torch.tensor(np.array(p_pred), dtype=torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        grad_E, grad_nu, grad_rho = ctx.saved_tensors
        # grad_output 为标量，乘上 jax 计算的梯度
        # 注意：梯度现在形状为[1]，需要提取元素值
        dE   = grad_output * grad_E[0]
        dnu  = grad_output * grad_nu[0]
        drho = grad_output * grad_rho[0]
        return dE, dnu, drho

def jax_fem_pressure(E, nu, rho):
    return JaxFEMMicPressure.apply(E, nu, rho)

# -------------------------
# 3. PyTorch 优化环节
# -------------------------
# 定义待优化材料参数，初始值来自工厂经验
E_init   = 3.0e9       # Young's modulus, Pa
nu_init  = 0.35        # Poisson ratio
rho_init = 1400.0      # density, kg/m³

# 作为 torch.nn.Parameter
E_param   = torch.nn.Parameter(torch.tensor(E_init, dtype=torch.float32))
nu_param  = torch.nn.Parameter(torch.tensor(nu_init, dtype=torch.float32))
rho_param = torch.nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))

# 优化器
optimizer = torch.optim.Adam([E_param, nu_param, rho_param], lr=5e-2)

# -------------------------
# 4. 优化循环
# -------------------------
n_epochs = 50
print("开始材料参数优化...")
start_time = time.time()
for epoch in range(n_epochs):
    optimizer.zero_grad()
    # 调用自定义函数，得到 FEM 模型预测的远端声压
    pred_p = jax_fem_pressure(E_param, nu_param, rho_param)
    print(f"pred_p: {pred_p}")
    # 定义 loss = (pred - meas)^2 （标量误差）
    loss = (pred_p - torch.tensor(meas_p, dtype=torch.float32))**2
    loss.backward()
    # 保证泊松比在 (0,0.5)
    with torch.no_grad():
        nu_param.clamp_(0.0, 0.4999)
    optimizer.step()
    if epoch % 1 == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch:03d}, Loss={loss.item():.6e}, E={E_param.item():.3e}, nu={nu_param.item():.4f}, rho={rho_param.item():.2f}, Time={elapsed:.2f}s")
print("优化完成！")
print(f"最终结果: E = {E_param.item():.3e} Pa, nu = {nu_param.item():.4f}, rho = {rho_param.item():.2f} kg/m³")
