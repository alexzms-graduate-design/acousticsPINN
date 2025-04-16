# jax_main.py
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
import numpy as np
import argparse
from jax_fem import CoupledFEMSolver

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='流固耦合FEM求解器 (JAX版本)')
parser.add_argument('--mode', type=str, default='optimize', 
                    choices=['optimize', 'single_run'],
                    help='运行模式: optimize进行优化, single_run只运行一次')
parser.add_argument('--volume_source', type=float, default=0.1,
                    help='体激励项强度 (Pa)')
args = parser.parse_args()

print(f"使用JAX版本 {jax.__version__}")
print(f"设备: {jax.devices()}")

# -------------------------------
# 设定基本参数
# -------------------------------
frequency = 10.0       # 噪音源频率 (Hz)
penalty = 1e8          # 耦合惩罚参数
mesh_file = "y_pipe.msh"   # 由 geometry_gmsh.py 生成的网格文件

# -------------------------------
# 定义体激励项
# -------------------------------
volume_source = args.volume_source    # 施加均匀的体激励项到整个计算域

# 定义空间变化的函数（如果需要的话）
def spatial_source(x, y, z):
    # 这里可以定义任意空间分布的激励函数
    # 如果z<0.1, 激励为1Pa, 否则为0Pa
    return volume_source if z < 0.01 else 0.0

print(f"[info] 使用空间变化的体激励项: {volume_source} Pa")

# -------------------------------
# 初始化 FEM 求解器
# -------------------------------
fem_solver = CoupledFEMSolver(mesh_file, frequency=frequency, cppenalty=penalty)

# -------------------------------
# 定义材料参数
# -------------------------------
E_init = 3.0e9      # 杨氏模量 (Pa)
nu_init = 0.35      # 泊松比
rho_init = 1400.0   # 固体密度 (kg/m³)

# 单次运行模式
if args.mode == 'single_run':
    print("[info] 以单次运行模式执行，使用固定参数")
    # 直接使用固定参数求解
    pred_pressure, u = fem_solver.solve(
        E_init,
        nu_init,
        rho_init,
        volume_source=spatial_source
    )
    print(f"[结果] 使用体激励项 {volume_source} Pa:")
    print(f"  杨氏模量 E = {E_init:.3e} Pa")
    print(f"  泊松比 nu = {nu_init:.4f}")
    print(f"  密度 rho_s = {rho_init:.2f} kg/m³")
    print(f"  麦克风处声压 = {float(pred_pressure):.6e} Pa")
    # 可以保存解向量u以供后处理
    # np.save('solution_with_volume_source.npy', np.array(u))
    exit(0)

# 优化模式 - 只有当mode='optimize'时才运行
print("[info] 以优化模式执行，优化材料参数")

# -------------------------------
# JAX优化实现
# -------------------------------
from jax import value_and_grad
import optax

# 初始化参数
params = {
    'E': E_init,
    'nu': nu_init,
    'rho_s': rho_init
}

# 定义损失函数
def loss_fn(params):
    E = params['E']
    nu = params['nu']
    rho_s = params['rho_s']
    
    # 约束泊松比在有效范围内
    nu = jnp.clip(nu, 0.0, 0.499)
    
    pred_pressure, _ = fem_solver.solve(E, nu, rho_s, volume_source=spatial_source)
    target_pressure = 0.5  # 目标麦克风声压值
    loss = (pred_pressure - target_pressure)**2
    
    return loss, (pred_pressure, E, nu, rho_s)

# 创建优化器
optimizer = optax.adam(learning_rate=5e-2)
opt_state = optimizer.init(params)

# 优化循环
n_epochs = 100

# 定义一个jit编译的更新函数
@jit
def update_step(params, opt_state):
    (loss, aux), grads = value_and_grad(loss_fn, has_aux=True)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    # 手动约束泊松比
    params['nu'] = jnp.clip(params['nu'], 0.0, 0.499)
    return params, opt_state, loss, aux

# 优化循环
print("[info] 开始优化")
for epoch in range(n_epochs):
    params, opt_state, loss, (pred_pressure, E, nu, rho_s) = update_step(params, opt_state)
    
    print(f"[info] 预测压力: {float(pred_pressure):.6e}, 目标压力: 0.5, 损失: {float(loss):.6e}")
    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d}: Loss = {float(loss):.6e}, Predicted Far Pressure = {float(pred_pressure):.6e}")
        print(f"   E = {float(E):.3e} Pa, nu = {float(nu):.4f}, rho_s = {float(rho_s):.2f} kg/m³")

print("优化结束。最终参数：")
print(f"  E = {float(params['E']):.3e} Pa")
print(f"  nu = {float(params['nu']):.4f}")
print(f"  rho_s = {float(params['rho_s']):.2f} kg/m³") 