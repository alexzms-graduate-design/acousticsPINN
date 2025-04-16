# numpy_main.py
import numpy as np
import argparse
from numpy_fem import CoupledFEMSolver

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='流固耦合FEM求解器 (NumPy版本)')
parser.add_argument('--mode', type=str, default='optimize', 
                    choices=['optimize', 'single_run'],
                    help='运行模式: optimize进行优化, single_run只运行一次')
parser.add_argument('--volume_source', type=float, default=0.1,
                    help='体激励项强度 (Pa)')
args = parser.parse_args()

print(f"Running with NumPy version: {np.__version__}")

# -------------------------------
# 设定基本参数
# -------------------------------
frequency = 10.0       # 噪音源频率 (Hz)
penalty = 1e8            # 耦合惩罚参数
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
    print(f"  麦克风处声压 = {pred_pressure:.6e} Pa")
    # 可以保存解向量u以供后处理
    # np.save('solution_with_volume_source.npy', u)
    exit(0)

# 优化模式 - 只有当mode='optimize'时才运行
print("[info] 以优化模式执行，优化材料参数")

# NumPy版本优化实现
# 初始化优化参数
params = {
    'E': E_init,
    'nu': nu_init,
    'rho_s': rho_init
}

# 定义损失函数
target_pressure = 0.5

def loss_fn(params):
    # NumPy版本不需要处理tracer问题
    pred_pressure = fem_solver.forward_numpy(
        params['E'], 
        params['nu'], 
        params['rho_s'], 
        volume_source=volume_source
    )
    return (pred_pressure - target_pressure)**2

# 简单的基于梯度下降的优化器
class SimpleOptimizer:
    def __init__(self, learning_rate=5e-2):
        self.learning_rate = learning_rate
        
    def step(self, params, grads):
        """根据梯度更新参数"""
        return {
            'E': params['E'] - self.learning_rate * grads['E'],
            'nu': np.clip(params['nu'] - self.learning_rate * grads['nu'], 0.0, 0.499),
            'rho_s': params['rho_s'] - self.learning_rate * grads['rho_s']
        }

# 创建优化器
optimizer = SimpleOptimizer(learning_rate=5e-2)

# 优化循环：根据远端麦克风处预测值与实测值的误差优化材料参数
print("[info] 开始优化")
n_epochs = 100

def optimization_step(params):
    # 使用数值微分计算梯度
    current_loss = loss_fn(params)
    
    # 计算数值梯度
    epsilon = 1e-4
    
    # 针对E的梯度
    params_E_plus = {**params, 'E': params['E'] + epsilon}
    params_E_minus = {**params, 'E': params['E'] - epsilon}
    grad_E = (loss_fn(params_E_plus) - loss_fn(params_E_minus)) / (2 * epsilon)
    
    # 针对nu的梯度
    params_nu_plus = {**params, 'nu': params['nu'] + epsilon}
    params_nu_minus = {**params, 'nu': params['nu'] - epsilon}
    grad_nu = (loss_fn(params_nu_plus) - loss_fn(params_nu_minus)) / (2 * epsilon)
    
    # 针对rho_s的梯度
    params_rho_plus = {**params, 'rho_s': params['rho_s'] + epsilon}
    params_rho_minus = {**params, 'rho_s': params['rho_s'] - epsilon}
    grad_rho = (loss_fn(params_rho_plus) - loss_fn(params_rho_minus)) / (2 * epsilon)
    
    # 构建梯度字典
    grads = {
        'E': grad_E,
        'nu': grad_nu,
        'rho_s': grad_rho
    }
    
    # 使用优化器更新参数
    updated_params = optimizer.step(params, grads)
    
    return updated_params, current_loss

for epoch in range(n_epochs):
    params, loss = optimization_step(params)
    
    # 输出当前进度
    if epoch % 1 == 0:
        # 计算用于报告的预测压力
        pred_pressure = fem_solver.forward_numpy(
            params['E'], params['nu'], params['rho_s'], volume_source=volume_source
        )
        
        print(f"Epoch {epoch:03d}: Loss = {loss:.6e}, Predicted Far Pressure = {pred_pressure:.6e}")
        print(f"   E = {params['E']:.3e} Pa, nu = {params['nu']:.4f}, rho_s = {params['rho_s']:.2f} kg/m³")

print("优化结束。最终参数：")
print(f"  E = {params['E']:.3e} Pa")
print(f"  nu = {params['nu']:.4f}")
print(f"  rho_s = {params['rho_s']:.2f} kg/m³") 