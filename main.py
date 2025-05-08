# main.py
import torch
import torch.optim as optim
import numpy as np
import argparse
from fem import CoupledFEMSolver
from torchviz import make_dot

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='流固耦合FEM求解器')
parser.add_argument('--mode', type=str, default='optimize', 
                    choices=['optimize', 'single_run'],
                    help='运行模式: optimize进行优化, single_run只运行一次')
parser.add_argument('--volume_source', type=float, default=0.1,
                    help='体激励项强度 (Pa)')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

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
fem_solver = CoupledFEMSolver(mesh_file, frequency=frequency, cppenalty=penalty).to(device)

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
        torch.tensor(E_init, dtype=torch.float32, device=device),
        torch.tensor(nu_init, dtype=torch.float32, device=device),
        torch.tensor(rho_init, dtype=torch.float32, device=device),
        volume_source=spatial_source
    )
    print(f"[结果] 使用体激励项 {volume_source} Pa:")
    print(f"  杨氏模量 E = {E_init:.3e} Pa")
    print(f"  泊松比 nu = {nu_init:.4f}")
    print(f"  密度 rho_s = {rho_init:.2f} kg/m³")
    print(f"  麦克风处声压 = {pred_pressure.item():.6e} Pa")
    # 可以保存解向量u以供后处理
    # torch.save(u, 'solution_with_volume_source.pt')
    exit(0)

# 优化模式 - 只有当mode='optimize'时才运行
print("[info] 以优化模式执行，优化材料参数")
E_param = torch.nn.Parameter(torch.tensor(E_init, dtype=torch.float32, device=device))
nu_param = torch.nn.Parameter(torch.tensor(nu_init, dtype=torch.float32, device=device))
rho_param = torch.nn.Parameter(torch.tensor(rho_init, dtype=torch.float32, device=device))

optimizer = optim.Adam([E_param, nu_param, rho_param], lr=5e-2)

# -------------------------------
# 设定目标：远端麦克风测量信号（例如 0.5 Pa）
target_pressure = torch.tensor(0.5, dtype=torch.float32, device=device)

# -------------------------------
# 优化循环：根据远端麦克风处预测值与实测值的误差优化材料参数
# -------------------------------
print("[info] 开始优化")
n_epochs = 100
for epoch in range(n_epochs):
    optimizer.zero_grad()
    pred_pressure, _ = fem_solver.solve(E_param, nu_param, rho_param, volume_source=spatial_source)
    loss = (pred_pressure - target_pressure)**2
    print(f"[info] 预测压力: {pred_pressure.item():.6e}, 目标压力: {target_pressure.item():.6e}, 损失: {loss.item():.6e}")
    # add backward computation graph visualization
    if epoch == 0:
        # set max recursion depth in python
        import sys
        sys.setrecursionlimit(100000)
        # Visualize the computation graph for the first epoch
        graph_params = {
            'E_param': E_param,
            'nu_param': nu_param,
            'rho_param': rho_param
        }
        # Pass parameters that require gradients to make_dot
        # We are interested in how loss flows back to these parameters
        dot = make_dot(loss, params=graph_params)
        dot.render("computation_graph", format="png") # Saves computation_graph.png
        print("[info] Computation graph saved to computation_graph.png")
    input("Press Enter to continue...")
    loss.backward()
    optimizer.step()
    # 保证泊松比在 (0, 0.5) 内
    with torch.no_grad():
        nu_param.clamp_(0.0, 0.499)
    if epoch % 1 == 0:
        print(f"Epoch {epoch:03d}: Loss = {loss.item():.6e}, Predicted Far Pressure = {pred_pressure.item():.6e}")
        print(f"   E = {E_param.item():.3e} Pa, nu = {nu_param.item():.4f}, rho_s = {rho_param.item():.2f} kg/m³")

print("优化结束。最终参数：")
print(f"  E = {E_param.item():.3e} Pa")
print(f"  nu = {nu_param.item():.4f}")
print(f"  rho_s = {rho_param.item():.2f} kg/m³")
