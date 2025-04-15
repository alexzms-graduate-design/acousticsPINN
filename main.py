# main.py
import torch
import torch.optim as optim
import numpy as np
from fem import CoupledFEMSolver

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Running on device: {device}")

# -------------------------------
# 设定基本参数
# -------------------------------
frequency = 10.0       # 噪音源频率 (Hz)
penalty = 1e8            # 耦合惩罚参数
mesh_file = "y_pipe.msh"   # 由 geometry_gmsh.py 生成的网格文件

# -------------------------------
# 初始化 FEM 求解器
# -------------------------------
fem_solver = CoupledFEMSolver(mesh_file, frequency=frequency, cppenalty=penalty).to(device)

# -------------------------------
# 定义材料参数（作为待优化变量）
# 初始值：E = 3e9 Pa, nu = 0.35, rho_s = 1400 kg/m³
E_init = 3.0e9
nu_init = 0.35
rho_init = 1400.0

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
    pred_pressure, _ = fem_solver.solve(E_param, nu_param, rho_param)
    loss = (pred_pressure - target_pressure)**2
    print(f"[info] 预测压力: {pred_pressure.item():.6e}, 目标压力: {target_pressure.item():.6e}, 损失: {loss.item():.6e}")
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
