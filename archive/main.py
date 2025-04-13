import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# -------------------------------
# 全局设置与几何参数
# -------------------------------
torch.manual_seed(0)
np.random.seed(0)

# 检查CUDA是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 时间区间 (s)
T_sim = 0.005  # 模拟时间范围（例如 5ms 内的稳态响应）
# 主管参数
x_main_min, x_main_max = 0.0, 1.5       # x 方向范围（m）
R_inner_main = 0.045  # 主管内半径 (m)
R_outer_main = 0.05   # 主管外半径 (m)
# 分支参数
branch_origin = np.array([0.5, 0.0, 0.0])  # 分支起点（主管上，x=0.5）
L_branch = 0.3         # 分支长度 (m)
# 取 45°向后插入：令分支中心轴方向为 d = (-cos45, sin45, 0)
d_branch = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])
R_inner_branch = R_inner_main
R_outer_branch = R_outer_main

# 麦克风位置（示例取主管中心线处，x=1.0处）
mic_pos = np.array([1.0, 0.0, 0.0])

# 声学参数（空气，已知常量）
rho_f = 1.2      # kg/m^3
c_f   = 343.0    # m/s

# 开放边界衰减参数（示例值）
alpha_open = 5.0

# -------------------------------
# 采样辅助函数
# -------------------------------

def sample_cylinder(domain_x, r_min, r_max, n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在一个圆柱体区域采样点（适用于主管液体域或结构域）
      - 轴向：x in [domain_x[0], domain_x[1]]
      - 横截面：半径 r in [r_min, r_max]，角度 theta in [0, 2pi)
      - 时间：t in [t_min, t_max]（如果 include_time=True）
    返回 tensor [n_points, 4]  （x,y,z,t）
    """
    x_vals = np.random.uniform(domain_x[0], domain_x[1], (n_points, 1))
    # 为保证均匀采样，使用反变换法采样 r^2
    r_sq = np.random.uniform(r_min**2, r_max**2, (n_points, 1))
    r_vals = np.sqrt(r_sq)
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    y_vals = r_vals * np.cos(theta)
    z_vals = r_vals * np.sin(theta)
    if include_time:
        t_vals = np.random.uniform(t_min, t_max, (n_points, 1))
        pts = np.concatenate([x_vals, y_vals, z_vals, t_vals], axis=1)
    else:
        pts = np.concatenate([x_vals, y_vals, z_vals], axis=1)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_branch_cylinder(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在分支内腔（流体域）采样：
      - 分支采用局部坐标： s 在 [0, L_branch], r in [0, R_inner_branch], theta in [0,2pi)
      - 全局坐标：  point = branch_origin + s*d_branch + r*(cos(theta)*u + sin(theta)*v)
      - 其中，u, v 为构成分支截面的正交基，取 u = (cos45, cos45, 0), v = (0, 0, 1)
    返回 tensor [n_points, 4]
    """
    s_vals = np.random.uniform(0, L_branch, (n_points, 1))
    # r in [0, R_inner_branch]
    r_sq = np.random.uniform(0, R_inner_branch**2, (n_points, 1))
    r_vals = np.sqrt(r_sq)
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    # 定义局部基底：对于 d_branch = (-1/sqrt2, 1/sqrt2, 0)
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])  # 与 d_branch 垂直
    v = np.array([0, 0, 1])  # 保证垂直 u 和 d_branch
    pts = []
    for i in range(n_points):
        s = s_vals[i,0]
        r = r_vals[i,0]
        th = theta[i,0]
        offset = r*(np.cos(th)*u + np.sin(th)*v)
        point = branch_origin + s*d_branch + offset
        pts.append(np.concatenate([point, 
                                    [np.random.uniform(t_min, t_max)]], axis=0))
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_annular_cylinder(domain_x, r_min, r_max, n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在一个圆柱形管壁区域（环形）采样：x in domain_x, r in [r_min, r_max]（采样在横截面上）。
    返回 tensor [n_points, 4]
    """
    return sample_cylinder(domain_x, r_min, r_max, n_points, include_time, t_min, t_max)

def sample_branch_annular(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在分支管壁区域（环形）采样
    """
    # 使用分支内腔和外腔的半径范围： r in [R_inner_branch, R_outer_branch]
    s_vals = np.random.uniform(0, L_branch, (n_points, 1))
    r_sq = np.random.uniform(R_inner_branch**2, R_outer_branch**2, (n_points, 1))
    r_vals = np.sqrt(r_sq)
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    v = np.array([0, 0, 1])
    pts = []
    for i in range(n_points):
        s = s_vals[i,0]
        r = r_vals[i,0]
        th = theta[i,0]
        offset = r*(np.cos(th)*u + np.sin(th)*v)
        point = branch_origin + s*d_branch + offset
        pts.append(np.concatenate([point, 
                                    [np.random.uniform(t_min, t_max)]], axis=0))
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_interface_main(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在主管内表面（流固界面）采样，即 r = R_inner_main，在截面上均匀采样。
    同时返回采样点及对应的外法向量（朝向管壁外侧，即沿 (0, y, z)/r）
    返回：x_if (tensor [N,4]) 和 normal (tensor [N,3])
    """
    x_vals = np.random.uniform(x_main_min, x_main_max, (n_points, 1))
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    r = R_inner_main * np.ones((n_points, 1))
    y_vals = r * np.cos(theta)
    z_vals = r * np.sin(theta)
    if include_time:
        t_vals = np.random.uniform(t_min, t_max, (n_points, 1))
        pts = np.concatenate([x_vals, y_vals, z_vals, t_vals], axis=1)
    else:
        pts = np.concatenate([x_vals, y_vals, z_vals], axis=1)
    # 外法向量： [0, cos(theta), sin(theta)]
    normals = np.concatenate([np.zeros_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(normals, dtype=torch.float32).to(device)

def sample_interface_branch(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在分支内表面采样（流固界面），即 r = R_inner_branch 在分支局部采样。
    同时返回全局坐标和法向量（法向量指向壁外侧，须与局部坐标一致）。
    利用分支参数，此处保存法向量时直接用采样时已知的局部 polar 信息。
    """
    s_vals = np.random.uniform(0, L_branch, (n_points, 1))
    r = R_inner_branch * np.ones((n_points, 1))
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    # 分支局部正交基
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    v = np.array([0, 0, 1])
    pts = []
    normals = []
    for i in range(n_points):
        s = s_vals[i,0]
        th = theta[i,0]
        offset = r[i,0]*(np.cos(th)*u + np.sin(th)*v)
        # 点的位置
        point = branch_origin + s*d_branch + offset
        # 局部法向量（在分支截面内）： (cos(th)*u + sin(th)*v)
        n_local = np.cos(th)*u + np.sin(th)*v
        if include_time:
            t_val = np.random.uniform(t_min, t_max)
            pt = np.concatenate([point, [t_val]], axis=0)
        else:
            pt = point
        pts.append(pt)
        normals.append(n_local)
    pts = np.array(pts)
    normals = np.array(normals)
    return torch.tensor(pts, dtype=torch.float32).to(device), torch.tensor(normals, dtype=torch.float32).to(device)

def sample_open_boundary_main(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在主管出口 x = x_main_max 处采样（开放边界）
    """
    x_vals = x_main_max * np.ones((n_points, 1))
    # 在管道截面内均匀采样，采用极坐标采样（半径在 [0, R_inner_main] 内）
    r_sq = np.random.uniform(0, R_inner_main**2, (n_points, 1))
    r_vals = np.sqrt(r_sq)
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    y_vals = r_vals * np.cos(theta)
    z_vals = r_vals * np.sin(theta)
    if include_time:
        t_vals = np.random.uniform(t_min, t_max, (n_points, 1))
        pts = np.concatenate([x_vals, y_vals, z_vals, t_vals], axis=1)
    else:
        pts = np.concatenate([x_vals, y_vals, z_vals], axis=1)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_open_boundary_branch(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在分支出口，即 s = L_branch 处采样（开放边界）
    """
    s_vals = L_branch * np.ones((n_points, 1))
    r_sq = np.random.uniform(0, R_inner_branch**2, (n_points, 1))
    r_vals = np.sqrt(r_sq)
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    v = np.array([0, 0, 1])
    pts = []
    for i in range(n_points):
        offset = r_vals[i,0]*(np.cos(theta[i,0])*u + np.sin(theta[i,0])*v)
        point = branch_origin + L_branch*d_branch + offset
        if include_time:
            t_val = np.random.uniform(t_min, t_max)
            pt = np.concatenate([point, [t_val]], axis=0)
        else:
            pt = point
        pts.append(pt)
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_mic_points(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在麦克风位置附近采样，麦克风位置固定于 mic_pos。
    这里假设采样点为 mic_pos 的位置，时间上分布采样
    """
    pts = []
    for i in range(n_points):
        t_val = np.random.uniform(t_min, t_max) if include_time else 0.0
        pt = np.concatenate([mic_pos, [t_val]], axis=0)
        pts.append(pt)
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

# -------------------------------
# 物理网络定义：流体域与结构域
# -------------------------------

class FluidNet(nn.Module):
    def __init__(self, layers):
        """
        输入： (x,y,z,t) ; 输出： p
        """
        super(FluidNet, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers)-1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.net = nn.ModuleList(layer_list)
        
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.net):
            out = layer(out)
            if i < len(self.net)-1:
                out = self.activation(out)
        return out  # 输出 shape [N,1]

class StructNet(nn.Module):
    def __init__(self, layers):
        """
        输入： (x,y,z,t) ; 输出： (u_x, u_y, u_z)
        """
        super(StructNet, self).__init__()
        self.activation = nn.Tanh()
        layer_list = []
        for i in range(len(layers)-1):
            layer_list.append(nn.Linear(layers[i], layers[i+1]))
        self.net = nn.ModuleList(layer_list)
        
    def forward(self, x):
        out = x
        for i, layer in enumerate(self.net):
            out = layer(out)
            if i < len(self.net)-1:
                out = self.activation(out)
        return out  # 输出 shape [N,3]

# -------------------------------
# 耦合 PINN 主类：含材料参数（PVC）及 PDE 残差构造
# -------------------------------

class CoupledPINN(nn.Module):
    def __init__(self, fluid_layers, struct_layers):
        super(CoupledPINN, self).__init__()
        self.fluid_net = FluidNet(fluid_layers)
        self.struct_net = StructNet(struct_layers)
        
        # PVC 材料待优化参数（初值参考文献）
        # 使用对数形式表示杨氏模量，便于优化 (log(3e9) ≈ 21.82)
        self.log_E = nn.Parameter(torch.tensor(21.82, dtype=torch.float32))  # log(E)
        self.nu = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))      # 泊松比
        
        # 定义密度可能的范围并归一化
        self.rho_min, self.rho_max = 500.0, 2500.0  # 密度范围 (kg/m^3)
        # 将1400归一化到[0,1]区间: (1400-500)/(2500-500) = 0.45
        self.rho_s_norm = nn.Parameter(torch.tensor(0.45, dtype=torch.float32))  # 归一化密度
        
        # 空气参数（常数）
        self.rho_f = rho_f
        self.c = c_f

    @property
    def E(self):
        # 从对数参数转回实际的杨氏模量
        return torch.exp(self.log_E)

    @property
    def rho_s(self):
        # 从归一化参数转回实际密度值
        # 确保归一化值在[0,1]区间
        rho_norm_clamped = torch.clamp(self.rho_s_norm, 0.0, 1.0)
        return self.rho_min + rho_norm_clamped * (self.rho_max - self.rho_min)

    def p_fluid(self, x):
        return self.fluid_net(x)
    
    def u_struct(self, x):
        return self.struct_net(x)
    
    def pde_fluid_residual(self, x):
        """
        声波方程： p_tt - c^2 * (p_xx + p_yy + p_zz) = 0
        """
        x.requires_grad_(True)
        p = self.p_fluid(x)
        grad_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                     create_graph=True, retain_graph=True)[0]
        p_t = grad_p[:, 3:4]
        # 二阶时间导数
        p_tt = torch.autograd.grad(p_t, x, grad_outputs=torch.ones_like(p_t),
                                   create_graph=True, retain_graph=True)[0][:, 3:4]
        # 二阶空间导数
        p_x = grad_p[:, 0:1]
        p_y = grad_p[:, 1:2]
        p_z = grad_p[:, 2:3]
        p_xx = torch.autograd.grad(p_x, x, grad_outputs=torch.ones_like(p_x),
                                   create_graph=True, retain_graph=True)[0][:, 0:1]
        p_yy = torch.autograd.grad(p_y, x, grad_outputs=torch.ones_like(p_y),
                                   create_graph=True, retain_graph=True)[0][:, 1:2]
        p_zz = torch.autograd.grad(p_z, x, grad_outputs=torch.ones_like(p_z),
                                   create_graph=True, retain_graph=True)[0][:, 2:3]
        lap_p = p_xx + p_yy + p_zz
        res = p_tt - (self.c**2)*lap_p
        return res
    
    def pde_struct_residual(self, x):
        """
        线弹性方程： ρ_s * u_tt = div(σ)
        σ = λ trace(ε) I + 2μ ε, 其中 λ, μ 由 E, ν 得出
        ε = 0.5*(grad(u) + grad(u)^T)
        """
        x.requires_grad_(True)
        u = self.u_struct(x)  # shape [N,3]
        
        # 计算时间二阶导数 (对 t 求二阶导)
        u_t = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0][:,3:4]
        u_tt = torch.autograd.grad(u_t, x, grad_outputs=torch.ones_like(u_t),
                                   create_graph=True, retain_graph=True)[0][:,3:4]
        # 注意：这里只对 u 的每个分量单独计算 t 二阶导，实际应对每分量分别计算
        # 为简单起见，假设所有 u 分量均用同一时间坐标
        # 计算空间导数
        grads = []
        for i in range(3):
            u_i = u[:, i:i+1]
            grad_u_i = torch.autograd.grad(u_i, x,
                                           grad_outputs=torch.ones_like(u_i),
                                           create_graph=True, retain_graph=True)[0]
            grads.append(grad_u_i)
        # grads[i] shape: [N,4]；取空间部分 (columns 0-2)
        u_x = grads[0][:,0:1]
        u_y = grads[0][:,1:2]
        u_z = grads[0][:,2:3]
        v_x = grads[1][:,0:1]
        v_y = grads[1][:,1:2]
        v_z = grads[1][:,2:3]
        w_x = grads[2][:,0:1]
        w_y = grads[2][:,1:2]
        w_z = grads[2][:,2:3]
        
        # 计算应变分量
        eps_xx = u_x
        eps_yy = v_y
        eps_zz = w_z
        eps_xy = 0.5*(grads[0][:,1:2] + grads[1][:,0:1])
        eps_xz = 0.5*(grads[0][:,2:3] + grads[2][:,0:1])
        eps_yz = 0.5*(grads[1][:,2:3] + grads[2][:,1:2])
        
        trace_eps = eps_xx + eps_yy + eps_zz
        
        # 拉梅参数
        lam = (self.E * self.nu) / ((1. + self.nu)*(1. - 2.*self.nu))
        mu = self.E / (2.0*(1. + self.nu))
        
        # 应力分量
        sigma_xx = lam*trace_eps + 2.*mu*eps_xx
        sigma_yy = lam*trace_eps + 2.*mu*eps_yy
        sigma_zz = lam*trace_eps + 2.*mu*eps_zz
        sigma_xy = 2.*mu*eps_xy
        sigma_xz = 2.*mu*eps_xz
        sigma_yz = 2.*mu*eps_yz
        
        # 计算 div(sigma) 分量：对每个应力分量对对应方向求导
        grad_sigma_xx = torch.autograd.grad(sigma_xx, x, grad_outputs=torch.ones_like(sigma_xx),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_xy = torch.autograd.grad(sigma_xy, x, grad_outputs=torch.ones_like(sigma_xy),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_xz = torch.autograd.grad(sigma_xz, x, grad_outputs=torch.ones_like(sigma_xz),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_yy = torch.autograd.grad(sigma_yy, x, grad_outputs=torch.ones_like(sigma_yy),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_yz = torch.autograd.grad(sigma_yz, x, grad_outputs=torch.ones_like(sigma_yz),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_zz = torch.autograd.grad(sigma_zz, x, grad_outputs=torch.ones_like(sigma_zz),
                                            create_graph=True, retain_graph=True)[0]
        
        div_sigma_x = grad_sigma_xx[:,0:1] + grad_sigma_xy[:,1:2] + grad_sigma_xz[:,2:3]
        div_sigma_y = grad_sigma_xy[:,0:1] + grad_sigma_yy[:,1:2] + grad_sigma_yz[:,2:3]
        div_sigma_z = grad_sigma_xz[:,0:1] + grad_sigma_yz[:,1:2] + grad_sigma_zz[:,2:3]
        
        # 结构 PDE 残差： ρ_s * u_tt - div(σ) = 0, 针对每个分量
        res_x = self.rho_s * u_tt - div_sigma_x
        res_y = self.rho_s * u_tt - div_sigma_y
        res_z = self.rho_s * u_tt - div_sigma_z
        
        # 这里只计算了时间二阶导的一个近似残差（实际中应对每个分量分别计算 u_tt），
        # 为简单起见，这里取残差均值
        residual = torch.cat([res_x, res_y, res_z], dim=1)
        return residual

    def interface_residual(self, x_if, normals):
        """
        交界面残差：
         - (i) 应力/声压连续： -p = n^T σ n
         - (ii) 速度连续： 流体法向速度与结构法向速度相等
        x_if: [N,4] 流固界面采样点
        normals: [N,3] 已知（单位法向量，指向管壁外侧）
        """
        x_if.requires_grad_(True)
        # 流体侧声压
        p_val = self.p_fluid(x_if)  # [N,1]
        # 计算结构侧应力（这里以结构网络输出近似求得，应按前面结构 PDE 计算）
        # 此处重新计算 u 与其梯度（也可封装成函数）
        u_val = self.u_struct(x_if)  # [N,3]
        grads_u = torch.autograd.grad(u_val, x_if, grad_outputs=torch.ones_like(u_val),
                                      create_graph=True, retain_graph=True)[0]  # [N,4,3]（各分量的梯度）
        # 计算应变（只取空间部分）
        # 为简化这里不做完整应力计算，而以 placeholder 代替（工业级实现中应精确计算）
        sigma_nn = torch.zeros_like(p_val)  # 替代写法
        # (i) 应力连续残差
        res_stress = -p_val - sigma_nn  # [N,1]
        
        # (ii) 速度连续残差：
        # 流体侧：法向速度 = (1/(rho_f*c)) * dp/dn
        p_grad = torch.autograd.grad(p_val, x_if, grad_outputs=torch.ones_like(p_val),
                                     create_graph=True, retain_graph=True)[0]  # [N,4]
        # 点乘法向量（取空间部分）
        dp_dn = (p_grad[:,0:3] * normals).sum(dim=1, keepdim=True)
        v_n_fluid = dp_dn / (self.rho_f * self.c)
        
        # 结构侧：法向速度 = d(u·n)/dt
        u_n = (u_val * normals).sum(dim=1, keepdim=True)
        u_n_t = torch.autograd.grad(u_n, x_if, grad_outputs=torch.ones_like(u_n),
                                    create_graph=True, retain_graph=True)[0][:,3:4]
        v_n_struct = u_n_t
        res_velocity = v_n_fluid - v_n_struct  # [N,1]
        
        # 合并界面残差
        res = torch.cat([res_stress, res_velocity], dim=1)
        return res

    def open_boundary_residual(self, x_open):
        """
        开放边界条件残差：
         采用： dp/dn + alpha*p = 0
         其中 n 为外法向量，对于主管出口， n = (1,0,0)
        """
        x_open.requires_grad_(True)
        p_val = self.p_fluid(x_open)  # [N,1]
        grad_p = torch.autograd.grad(p_val, x_open, grad_outputs=torch.ones_like(p_val),
                                     create_graph=True, retain_graph=True)[0]
        # 对于主管出口，法向量 n = (1,0,0)，则 dp/dn = dp/dx
        dp_dn = grad_p[:, 0:1]
        res = dp_dn + alpha_open * p_val
        return res

    def data_residual(self, x_data, p_data):
        """
        数据约束：要求在麦克风位置，网络预测声压与测量值（这里用正弦波模拟）接近
        """
        p_pred = self.p_fluid(x_data)
        return p_pred - p_data

    def total_loss(self, data):
        """
        data: dict, 包含各类采样点及相应数据
         - x_f: 流体域 collocation 点 [N_f,4]
         - x_s: 结构域 collocation 点 [N_s,4]
         - x_if, n_if: 流固界面点与其法向量 [N_if,4] 和 [N_if,3]
         - x_ob: 开放边界点 [N_ob,4]
         - x_mic, p_mic: 麦克风数据点与声压 [N_mic,4] 和 [N_mic,1]
        """
        x_f = data['x_f']
        x_s = data['x_s']
        x_if = data['x_if']
        n_if = data['n_if']
        x_ob = data['x_ob']
        x_mic = data['x_mic']
        p_mic = data['p_mic']
        
        res_f = self.pde_fluid_residual(x_f)
        loss_f = torch.mean(res_f**2)
        
        res_s = self.pde_struct_residual(x_s)
        loss_s = torch.mean(res_s**2)
        
        res_if = self.interface_residual(x_if, n_if)
        loss_if = torch.mean(res_if**2)
        
        res_ob = self.open_boundary_residual(x_ob)
        loss_ob = torch.mean(res_ob**2)
        
        res_mic = self.data_residual(x_mic, p_mic)
        loss_mic = torch.mean(res_mic**2)
        
        total_loss = loss_f + loss_s + loss_if + loss_ob + loss_mic
        return total_loss, {'loss_f': loss_f, 'loss_s': loss_s,
                              'loss_if': loss_if, 'loss_ob': loss_ob, 'loss_mic': loss_mic}

# -------------------------------
# 构造采样点数据集
# -------------------------------

N_f_main  = 1000
N_f_branch = 300
N_s_main  = 1000
N_s_branch = 300
N_if_main = 300
N_if_branch = 100
N_ob_main = 200
N_ob_branch = 50
N_mic = 50

# 流体域：主管与分支
x_f_main = sample_cylinder(domain_x=[x_main_min, x_main_max], r_min=0.0, r_max=R_inner_main, n_points=N_f_main)
x_f_branch = sample_branch_cylinder(N_f_branch)
x_f = torch.cat([x_f_main, x_f_branch], dim=0)

# 结构域：主管壁（环形）与分支壁
x_s_main = sample_annular_cylinder(domain_x=[x_main_min, x_main_max], r_min=R_inner_main, r_max=R_outer_main, n_points=N_s_main)
x_s_branch = sample_branch_annular(N_s_branch)
x_s = torch.cat([x_s_main, x_s_branch], dim=0)

# 流固界面：分别采样主管内表面与分支内表面
x_if_main, n_if_main = sample_interface_main(N_if_main)
x_if_branch, n_if_branch = sample_interface_branch(N_if_branch)
x_if = torch.cat([x_if_main, x_if_branch], dim=0)
n_if = torch.cat([n_if_main, n_if_branch], dim=0)

# 开放边界：主管出口与分支出口
x_ob_main = sample_open_boundary_main(N_ob_main)
x_ob_branch = sample_open_boundary_branch(N_ob_branch)
x_ob = torch.cat([x_ob_main, x_ob_branch], dim=0)

# 麦克风数据：假设在 mic_pos 处，声压为正弦波，频率 f = 1000Hz, 振幅 1.0 Pa
f0 = 1000  # Hz
def mic_signal(t):
    return np.sin(2*np.pi*f0*t)

x_mic = sample_mic_points(N_mic)
# 取麦克风点的时间坐标 (最后一列) 计算 p_mic
t_mic = x_mic[:,3].detach().cpu().numpy().flatten()
p_mic_np = mic_signal(t_mic)
p_mic = torch.tensor(p_mic_np.reshape(-1,1), dtype=torch.float32).to(device)

# -------------------------------
# 可视化管道结构
# -------------------------------
def visualize_pipe_structure():
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 将张量转换为numpy数组，仅保留空间坐标
    f_main_np = x_f_main.cpu().numpy()[:, 0:3]
    f_branch_np = x_f_branch.cpu().numpy()[:, 0:3]
    s_main_np = x_s_main.cpu().numpy()[:, 0:3]
    s_branch_np = x_s_branch.cpu().numpy()[:, 0:3]
    if_main_np = x_if_main.cpu().numpy()[:, 0:3]
    if_branch_np = x_if_branch.cpu().numpy()[:, 0:3]
    ob_main_np = x_ob_main.cpu().numpy()[:, 0:3]
    ob_branch_np = x_ob_branch.cpu().numpy()[:, 0:3]
    
    # 绘制主管流体域
    ax.scatter(f_main_np[:, 0], f_main_np[:, 1], f_main_np[:, 2], 
               color='blue', alpha=0.1, s=10, label='Main Pipe Fluid')
    
    # 绘制分支流体域
    ax.scatter(f_branch_np[:, 0], f_branch_np[:, 1], f_branch_np[:, 2], 
               color='cyan', alpha=0.3, s=10, label='Branch Pipe Fluid')
    
    # 绘制主管结构域
    ax.scatter(s_main_np[:, 0], s_main_np[:, 1], s_main_np[:, 2], 
               color='gray', alpha=0.5, s=10, label='Main Pipe Wall')
    
    # 绘制分支结构域
    ax.scatter(s_branch_np[:, 0], s_branch_np[:, 1], s_branch_np[:, 2], 
               color='darkgray', alpha=0.5, s=10, label='Branch Pipe Wall')
    
    # 绘制主管流固界面
    ax.scatter(if_main_np[:, 0], if_main_np[:, 1], if_main_np[:, 2], 
               color='green', alpha=0.8, s=15, label='Main Interface')
    
    # 绘制分支流固界面
    ax.scatter(if_branch_np[:, 0], if_branch_np[:, 1], if_branch_np[:, 2], 
               color='lime', alpha=0.8, s=15, label='Branch Interface')
    
    # 绘制开放边界
    ax.scatter(ob_main_np[:, 0], ob_main_np[:, 1], ob_main_np[:, 2], 
               color='red', alpha=0.8, s=20, label='Main Outlet')
    ax.scatter(ob_branch_np[:, 0], ob_branch_np[:, 1], ob_branch_np[:, 2], 
               color='orange', alpha=0.8, s=20, label='Branch Outlet')
    
    # 绘制麦克风位置
    ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], 
               color='black', s=100, marker='*', label='Microphone')
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Pipe System Sampling Points')
    
    # 设置坐标轴范围
    ax.set_xlim([x_main_min-0.1, x_main_max+0.1])
    ax.set_ylim([-0.4, 0.4])
    ax.set_zlim([-0.4, 0.4])
    
    # 显示图例
    ax.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.savefig('pipe_structure_points.png', dpi=300)
    plt.show()

def visualize_pipe_geometry():
    """绘制管道的几何结构线框图"""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 创建圆柱体和圆环函数
    def plot_cylinder(ax, x_range, radius, color, alpha=1.0, label=None, linestyle='-'):
        # 在指定x范围内绘制圆柱体
        theta = np.linspace(0, 2*np.pi, 100)
        x = np.linspace(x_range[0], x_range[1], 20)
        theta_grid, x_grid = np.meshgrid(theta, x)
        y_grid = radius * np.cos(theta_grid)
        z_grid = radius * np.sin(theta_grid)
        
        # 绘制圆柱体曲面
        ax.plot_surface(x_grid, y_grid, z_grid, color=color, alpha=alpha, shade=False)
        
        # 绘制端面圆环
        for x_pos in [x_range[0], x_range[1]]:
            circle_x = np.ones_like(theta) * x_pos
            circle_y = radius * np.cos(theta)
            circle_z = radius * np.sin(theta)
            ax.plot(circle_x, circle_y, circle_z, color=color, linestyle=linestyle)
        
        # 只在第一个圆环添加标签
        if label:
            ax.plot([x_range[0]], [radius], [0], color=color, label=label)
    
    # 绘制主管内外壁
    plot_cylinder(ax, [x_main_min, x_main_max], R_inner_main, 'blue', alpha=0.3, label='Main Pipe Inner Wall')
    plot_cylinder(ax, [x_main_min, x_main_max], R_outer_main, 'gray', alpha=0.3, label='Main Pipe Outer Wall')
    
    # 分支管几何计算和绘制
    # 分支内壁
    branch_end = branch_origin + L_branch * d_branch
    
    # 创建分支管的线框
    def plot_branch_cylinder(ax, start_point, direction, length, radius, color, alpha=0.3, label=None):
        # 沿着direction方向创建一个圆柱体，起点为start_point
        # 创建局部坐标系
        dir_norm = direction / np.linalg.norm(direction)
        # 创建垂直于dir_norm的基向量
        u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])  # 与分支方向垂直
        v = np.array([0, 0, 1])  # 垂直于u和dir_norm
        
        # 在分支轴上采样点
        s_vals = np.linspace(0, length, 20)
        # 在圆周上采样点
        theta = np.linspace(0, 2*np.pi, 30)
        
        # 创建圆柱体表面网格
        s_grid, theta_grid = np.meshgrid(s_vals, theta)
        points = []
        
        for i in range(s_grid.shape[0]):
            for j in range(s_grid.shape[1]):
                s = s_grid[i, j]
                th = theta_grid[i, j]
                # 计算分支上的点
                point = start_point + s * dir_norm + radius * (np.cos(th) * u + np.sin(th) * v)
                points.append(point)
        
        points = np.array(points).reshape(s_grid.shape[0], s_grid.shape[1], 3)
        
        # 绘制分支表面
        x_surface = points[:, :, 0]
        y_surface = points[:, :, 1]
        z_surface = points[:, :, 2]
        ax.plot_surface(x_surface, y_surface, z_surface, color=color, alpha=alpha, shade=False)
        
        # 绘制端口圆环
        for s in [0, length]:
            circle_points = []
            for th in theta:
                point = start_point + s * dir_norm + radius * (np.cos(th) * u + np.sin(th) * v)
                circle_points.append(point)
            circle_points = np.array(circle_points)
            ax.plot(circle_points[:, 0], circle_points[:, 1], circle_points[:, 2], color=color)
        
        # 只在起点添加标签
        if label:
            label_point = start_point + radius * u
            ax.plot([label_point[0]], [label_point[1]], [label_point[2]], color=color, label=label)
    
    # 绘制分支管内外壁
    plot_branch_cylinder(ax, branch_origin, d_branch, L_branch, R_inner_branch, 'cyan', alpha=0.3, label='Branch Inner Wall')
    plot_branch_cylinder(ax, branch_origin, d_branch, L_branch, R_outer_branch, 'darkgray', alpha=0.3, label='Branch Outer Wall')
    
    # 绘制麦克风位置
    ax.scatter(mic_pos[0], mic_pos[1], mic_pos[2], color='red', s=100, marker='*', label='Microphone')
    
    # 绘制声源位置（主管入口）
    # 创建声源图案（同心圆）
    theta = np.linspace(0, 2*np.pi, 100)
    source_x = np.ones_like(theta) * x_main_min
    for r_multiplier in [0.3, 0.6, 0.9]:
        r = R_inner_main * r_multiplier
        source_y = r * np.cos(theta)
        source_z = r * np.sin(theta)
        ax.plot(source_x, source_y, source_z, 'r-', linewidth=1.5)
    
    # 添加声源波浪线
    wave_x = np.linspace(x_main_min-0.1, x_main_min, 20)
    wave_y = np.zeros_like(wave_x)
    wave_z = 0.02 * np.sin(50 * wave_x)
    ax.plot(wave_x, wave_y, wave_z, 'r-', linewidth=2)
    ax.text(x_main_min-0.15, 0, 0, "Source", color='red', fontsize=12)
    
    # 设置坐标轴标签和标题
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_zlabel('Z (m)')
    ax.set_title('Pipe System Geometry with Source and Microphone')
    
    # 设置坐标轴范围
    ax.set_xlim([x_main_min-0.2, x_main_max+0.1])
    ax.set_ylim([-0.4, 0.4])
    ax.set_zlim([-0.4, 0.4])
    
    # 设置视角
    ax.view_init(elev=20, azim=-60)
    
    # 显示图例
    ax.legend()
    
    # 显示图形
    plt.tight_layout()
    plt.savefig('pipe_structure_geometry.png', dpi=300)
    plt.show()

# 在开始训练前可视化管道结构
print("可视化管道结构...")
visualize_pipe_structure()
visualize_pipe_geometry()

# -------------------------------
# 实例化模型、优化器与训练循环
# -------------------------------

fluid_layers  = [4, 64, 64, 64, 1]
struct_layers = [4, 64, 64, 64, 3]
model = CoupledPINN(fluid_layers, struct_layers)
model = model.to(device)  # 将模型迁移到CUDA

# 分开优化网络参数和材料参数
network_params = list(model.fluid_net.parameters()) + list(model.struct_net.parameters())
material_params = [model.log_E, model.nu, model.rho_s_norm]

# 为网络和材料参数使用不同的优化器
optimizer_network = optim.Adam(network_params, lr=1e-3)
optimizer_material = optim.Adam(material_params, lr=5e-2)  # 材料参数使用较大学习率

# 添加学习率调度器
scheduler_network = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_network, mode='min', factor=0.5, patience=500, verbose=True)
scheduler_material = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer_material, mode='min', factor=0.5, patience=300, verbose=True)

# 组合数据字典并将数据迁移到CUDA
data = {
    'x_f': x_f.to(device),
    'x_s': x_s.to(device),
    'x_if': x_if.to(device),
    'n_if': n_if.to(device),
    'x_ob': x_ob.to(device),
    'x_mic': x_mic.to(device),
    'p_mic': p_mic.to(device)
}

# 训练循环（可调迭代次数）
nIter = 5000
start_time = time.time()
last_time = start_time

# 用于保存最佳模型
best_loss = float('inf')
best_model_state = None
best_iter = -1

for it in range(nIter):
    optimizer_network.zero_grad()
    optimizer_material.zero_grad()
    
    loss, loss_dict = model.total_loss(data)
    loss.backward()
    
    # 在优化器更新前对泊松比进行约束（通常在[0, 0.5)区间）
    with torch.no_grad():
        model.nu.clamp_(0.0, 0.499)  # 泊松比约束
    
    optimizer_network.step()
    optimizer_material.step()
    
    # 检查并保存最佳模型
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = {
            'model': model.state_dict(),
            'optimizer_network': optimizer_network.state_dict(),
            'optimizer_material': optimizer_material.state_dict(),
            'loss': loss.item(),
            'loss_dict': {k: v.item() for k, v in loss_dict.items()},
            'iter': it,
            'params': {
                'E': model.E.item(),
                'nu': model.nu.item(),
                'rho_s': model.rho_s.item()
            }
        }
        best_iter = it
    
    # 每500次迭代更新学习率
    if it % 500 == 0:
        scheduler_network.step(loss)
        scheduler_material.step(loss)
        
        current_time = time.time()
        elapsed = current_time - start_time
        iter_time = current_time - last_time
        last_time = current_time
        
        print(f"Iter {it}: Total Loss = {loss.item():.4e}")
        print(f"  Fluid PDE Loss = {loss_dict['loss_f'].item():.4e}, Struct PDE Loss = {loss_dict['loss_s'].item():.4e}")
        print(f"  Interface Loss = {loss_dict['loss_if'].item():.4e}, Open BC Loss = {loss_dict['loss_ob'].item():.4e}, Data Loss = {loss_dict['loss_mic'].item():.4e}")
        print(f"  E = {model.E.item():.3e}, nu = {model.nu.item():.3f}, rho_s = {model.rho_s.item():.2f}")
        print(f"  LR network: {optimizer_network.param_groups[0]['lr']:.2e}, LR material: {optimizer_material.param_groups[0]['lr']:.2e}")
        print(f"  Time: {elapsed:.2f}s total, {iter_time:.2f}s for last 500 iterations ({iter_time/500*1000:.2f}ms/iter)")
        print(f"  Best iter: {best_iter}, Best loss: {best_loss:.4e}")
        print("-----------------------------------------------------")

# 训练结束后保存最佳模型
if best_model_state is not None:
    torch.save(best_model_state, 'best_model.pth')
    print(f"训练结束，保存最佳模型 (iter {best_iter}):")
    print(f"  最佳损失：{best_loss:.4e}")
    print(f"  优化参数：E = {best_model_state['params']['E']:.3e}, nu = {best_model_state['params']['nu']:.3f}, rho_s = {best_model_state['params']['rho_s']:.2f}")
