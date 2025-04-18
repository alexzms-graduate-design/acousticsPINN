# jax_main.py
import jax
import jax.numpy as jnp
import optax
import numpy as np
import argparse
from jax_fem import CoupledFEMSolver

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='流固耦合FEM求解器 (JAX版本)')
parser.add_argument('--mode', type=str, default='optimize', 
                    choices=['optimize', 'single_run'],
                    help='运行模式: optimize进行优化, single_run只运行一次')
parser.add_argument('--volume_source', type=float, default=1,
                    help='体激励项强度 (Pa)')
parser.add_argument('--target_pressure', type=float, default=0.5,
                    help='目标声压 (Pa)')
args = parser.parse_args()

# JAX的设备自动检测
device = jax.devices()[0]
print(f"Running on device: {device}")

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

# 或者使用空间变化的函数（如果需要的话可以取消注释）
def spatial_source(x, y, z):
    # 这里可以定义任意空间分布的激励函数
    # 如果z<0.1, 激励为1Pa, 否则为0Pa
    return volume_source if z < 0.01 else 0.0

print(f"[info] 使用空间变化的体激励项: {volume_source} Pa")

# -------------------------------
# 初始化 FEM 求解器
# -------------------------------
print("[info] 初始化求解器...")
fem_solver = CoupledFEMSolver(mesh_file, frequency=frequency, cppenalty=penalty)

# -------------------------------
# 定义材料参数
# -------------------------------
E_init = jnp.log(3.0e9) / 10.0      # 杨氏模量 (Pa)
nu_init = 0.35      # 泊松比
rho_init = jnp.log(1400.0)   # 固体密度 (kg/m³)

# 单次运行模式
if args.mode == 'single_run':
    print("[info] 以单次运行模式执行，使用固定参数")
    # 直接使用固定参数求解
    # 转换参数为JAX数组
    E = jnp.array(E_init, dtype=jnp.float32)
    nu = jnp.array(nu_init, dtype=jnp.float32)
    rho_s = jnp.array(rho_init, dtype=jnp.float32)
    
    pred_pressure, u = fem_solver.solve(
        E, nu, rho_s,
        volume_source=spatial_source
    )
    
    print(f"[结果] 使用体激励项 {volume_source} Pa:")
    print(f"  杨氏模量 E = {jnp.exp(10*E_init):.3e} Pa")
    print(f"  泊松比 nu = {nu_init:.4f}")
    print(f"  密度 rho_s = {jnp.exp(rho_init):.2f} kg/m³")
    print(f"  麦克风处声压 = {pred_pressure.item():.6e} Pa")
    # 可以保存解向量u以供后处理
    # np.save('solution_with_volume_source.npy', np.array(u))
    exit(0)

# 优化模式 - 只有当mode='optimize'时才运行
print("[info] 以优化模式执行，优化材料参数")

# JAX实现的优化与PyTorch不同，需要使用函数式编程风格
# 定义初始参数
params = {
    'E': jnp.array(E_init, dtype=jnp.float32),
    'nu': jnp.array(nu_init, dtype=jnp.float32),
    'rho_s': jnp.array(rho_init, dtype=jnp.float32)
}

# 设定目标：远端麦克风测量信号（例如 0.5 Pa）
target_pressure = jnp.array(args.target_pressure, dtype=jnp.float32)

# 定义损失函数
def loss_fn(params):
    E = params['E']
    nu = params['nu']
    rho_s = params['rho_s']
    
    # 确保参数在有效范围内
    nu = jnp.clip(nu, 0.0, 0.499)  # 泊松比限制
    
    # 求解并计算损失
    pred_pressure, _ = fem_solver.solve(E, nu, rho_s, volume_source=spatial_source)
    loss = (pred_pressure - target_pressure)**2
    
    return loss, pred_pressure

# 优化设置
optimizer = optax.adam(learning_rate=5e-2)
opt_state = optimizer.init(params)

# 定义更新函数
def update(params, opt_state):
    (loss, pred_pressure), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    print(f"[info] grads: {grads}")
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    
    # 手动裁剪参数范围
    params['nu'] = jnp.clip(params['nu'], 0.0, 0.499)
    
    return params, opt_state, loss, pred_pressure

# 优化循环
print("[info] 开始优化")
n_epochs = 100
for epoch in range(n_epochs):
    params, opt_state, loss, pred_pressure = update(params, opt_state)
    
    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d}: Loss = {loss.item():.6e}, Predicted Far Pressure = {pred_pressure.item():.6e}")
        print(f"   E = {jnp.exp(10*params['E'].item()):.3e} Pa, nu = {params['nu'].item():.4f}, rho_s = {jnp.exp(params['rho_s'].item()):.2f} kg/m³")

print("优化结束。最终参数：")
print(f"  E = {jnp.exp(10*params['E'].item()):.3e} Pa")
print(f"  nu = {params['nu'].item():.4f}")
print(f"  rho_s = {jnp.exp(params['rho_s'].item()):.2f} kg/m³")
