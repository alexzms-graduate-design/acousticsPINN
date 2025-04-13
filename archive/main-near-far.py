import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time

# ===============================
# 全局设置与几何/物理参数
# ===============================
torch.manual_seed(0)
np.random.seed(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 模拟时间范围（秒）
T_sim = 0.005  # 例如 5ms内响应

# 主管几何参数（沿 x 轴延伸）
x_main_min, x_main_max = 0.0, 1.5   # 主管长度 1.5m
R_inner_main = 0.045  # 内半径 4.5cm
R_outer_main = 0.05   # 外半径 5cm

# 分支参数：在主管 x=0.5 处以 45°向后延伸 0.3m
branch_origin = np.array([0.5, 0.0, 0.0])
L_branch = 0.3
d_branch = np.array([-1/np.sqrt(2), 1/np.sqrt(2), 0])  # 分支方向
R_inner_branch = R_inner_main
R_outer_branch = R_outer_main

# 麦克风位置
near_mic_pos = np.array([0.03, 0.0, 0.0])  # 近端（声源）位置
far_mic_pos  = np.array([1.0, 0.0, 0.0])   # 远端测量位置

# 空气参数（流体域）
rho_f = 1.2      # kg/m³
c_f   = 343.0    # m/s

# 开放边界衰减参数
alpha_open = 5.0

# -------------------------------
# 材料参数（管壁，PVC）初始值（预训练阶段固定）
# 预设：E=3e9 Pa，ν=0.35，ρₛ=1400 kg/m³
# 在优化阶段，这三个参数将作为可训练变量
E_init = 3e9
nu_init = 0.35
rho_s_init = 1400.0

# ===============================
# 采样辅助函数
# ===============================
def sample_cylinder(domain_x, r_min, r_max, n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在指定 x 范围的圆柱区域内均匀采样点，适用于主管内腔（流体域）或主管壁（环形）。
    返回 (x,y,z,t) 坐标，shape [n_points,4]
    """
    x_vals = np.random.uniform(domain_x[0], domain_x[1], (n_points, 1))
    # 均匀采样横截面时，应均匀采样 r^2
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
    在分支内腔采样。局部坐标：s in [0,L_branch]，r in [0,R_inner_branch]，θ in [0,2π).
    全局坐标： point = branch_origin + s*d_branch + offset.
    """
    s_vals = np.random.uniform(0, L_branch, (n_points, 1))
    r_sq = np.random.uniform(0, R_inner_branch**2, (n_points, 1))
    r_vals = np.sqrt(r_sq)
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    # 定义局部基：令 u=(cos45, cos45, 0), v=(0,0,1)
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    v = np.array([0, 0, 1])
    pts = []
    for i in range(n_points):
        s = s_vals[i,0]
        r = r_vals[i,0]
        th = theta[i,0]
        offset = r * (np.cos(th) * u + np.sin(th) * v)
        point = branch_origin + s * d_branch + offset
        pts.append(np.concatenate([point, [np.random.uniform(t_min, t_max)]], axis=0))
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_annular_cylinder(domain_x, r_min, r_max, n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在主管管壁（环形）区域内均匀采样，x 在给定区间内，r 在 [r_min, r_max]
    """
    return sample_cylinder(domain_x, r_min, r_max, n_points, include_time, t_min, t_max)

def sample_branch_annular(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在分支壁区域内采样，r in [R_inner_branch, R_outer_branch]
    """
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
        offset = r * (np.cos(th) * u + np.sin(th) * v)
        point = branch_origin + s*d_branch + offset
        pts.append(np.concatenate([point, [np.random.uniform(t_min, t_max)]], axis=0))
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

def sample_interface_main(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在主管内表面（流固界面）采样，固定 r = R_inner_main，
    同时返回外法向量 n = (0, cosθ, sinθ)
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
    normals = np.concatenate([np.zeros_like(theta), np.cos(theta), np.sin(theta)], axis=1)
    return (torch.tensor(pts, dtype=torch.float32).to(device),
            torch.tensor(normals, dtype=torch.float32).to(device))

def sample_interface_branch(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在分支内表面采样，固定 r = R_inner_branch，
    同时返回外法向量 n_local = (cosθ*u + sinθ*v)
    """
    s_vals = np.random.uniform(0, L_branch, (n_points, 1))
    r = R_inner_branch * np.ones((n_points, 1))
    theta = np.random.uniform(0, 2*np.pi, (n_points, 1))
    u = np.array([1/np.sqrt(2), 1/np.sqrt(2), 0])
    v = np.array([0, 0, 1])
    pts = []
    normals = []
    for i in range(n_points):
        s = s_vals[i,0]
        th = theta[i,0]
        offset = r[i,0] * (np.cos(th)*u + np.sin(th)*v)
        point = branch_origin + s * d_branch + offset
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
    return (torch.tensor(pts, dtype=torch.float32).to(device),
            torch.tensor(normals, dtype=torch.float32).to(device))

def sample_open_boundary_main(n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在主管出口（x = x_main_max）处采样（开放边界）
    """
    x_vals = x_main_max * np.ones((n_points, 1))
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

def sample_mic_at(pos, n_points, include_time=True, t_min=0.0, t_max=T_sim):
    """
    在指定位置 pos (如 near_mic_pos, far_mic_pos) 附近采样，固定空间位置。
    返回 [n_points,4]
    """
    pts = []
    for i in range(n_points):
        t_val = np.random.uniform(t_min, t_max) if include_time else 0.0
        pt = np.concatenate([pos, [t_val]], axis=0)
        pts.append(pt)
    pts = np.array(pts)
    return torch.tensor(pts, dtype=torch.float32).to(device)

# ===============================
# 物理网络：流体域与结构域
# ===============================
class FluidNet(nn.Module):
    def __init__(self, layers):
        """
        输入： (x, y, z, t) ; 输出： p
        """
        super(FluidNet, self).__init__()
        self.activation = nn.Tanh()
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.net(x)  # shape [N,1]

class StructNet(nn.Module):
    def __init__(self, layers):
        """
        输入： (x,y,z,t) ; 输出： (u_x, u_y, u_z)
        """
        super(StructNet, self).__init__()
        self.activation = nn.Tanh()
        modules = []
        for i in range(len(layers)-1):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                modules.append(nn.Tanh())
        self.net = nn.Sequential(*modules)
    
    def forward(self, x):
        return self.net(x)  # shape [N,3]

# ===============================
# Coupled PINN 定义
# ===============================
class CoupledPINN(nn.Module):
    def __init__(self, fluid_layers, struct_layers, fixed_material=False):
        """
        fixed_material: 若为 True，则材料参数在预训练阶段固定，不参与优化
        """
        super(CoupledPINN, self).__init__()
        self.fluid_net = FluidNet(fluid_layers)
        self.struct_net = StructNet(struct_layers)
        
        # 材料参数：以对数形式训练 E，泊松比 nu，密度用归一化参数
        # 预训练时可固定（fixed_material=True）后再在材料参数优化阶段解冻
        self.fixed_material = fixed_material
        if not fixed_material:
            self.log_E = nn.Parameter(torch.tensor(np.log(E_init), dtype=torch.float32))
            self.nu = nn.Parameter(torch.tensor(nu_init, dtype=torch.float32))
            self.rho_min, self.rho_max = 500.0, 2500.0
            self.rho_s_norm = nn.Parameter(torch.tensor((rho_s_init - 500.0)/(2500.0 - 500.0), dtype=torch.float32))
        else:
            # 固定材料参数，不作为参数（用 buffer保存）
            self.register_buffer('log_E', torch.tensor(np.log(E_init), dtype=torch.float32))
            self.register_buffer('nu', torch.tensor(nu_init, dtype=torch.float32))
            self.rho_min, self.rho_max = 500.0, 2500.0
            rho_norm = (rho_s_init - 500.0)/(2500.0 - 500.0)
            self.register_buffer('rho_s_norm', torch.tensor(rho_norm, dtype=torch.float32))
        
        self.rho_f = rho_f
        self.c = c_f
    
    @property
    def E(self):
        return torch.exp(self.log_E)
    
    @property
    def rho_s(self):
        rho_norm = torch.clamp(self.rho_s_norm, 0.0, 1.0)
        return self.rho_min + rho_norm*(self.rho_max - self.rho_min)
    
    def p_fluid(self, x):
        return self.fluid_net(x)
    
    def u_struct(self, x):
        return self.struct_net(x)
    
    def pde_fluid_residual(self, x):
        """
        声波方程： p_tt - c^2*(p_xx+p_yy+p_zz) = 0
        """
        x.requires_grad_(True)
        p = self.p_fluid(x)
        grad_p = torch.autograd.grad(p, x, torch.ones_like(p),
                                     create_graph=True, retain_graph=True)[0]
        p_t = grad_p[:,3:4]
        p_tt = torch.autograd.grad(p_t, x, torch.ones_like(p_t),
                                   create_graph=True, retain_graph=True)[0][:,3:4]
        p_x = grad_p[:,0:1]
        p_y = grad_p[:,1:2]
        p_z = grad_p[:,2:3]
        p_xx = torch.autograd.grad(p_x, x, torch.ones_like(p_x),
                                   create_graph=True, retain_graph=True)[0][:,0:1]
        p_yy = torch.autograd.grad(p_y, x, torch.ones_like(p_y),
                                   create_graph=True, retain_graph=True)[0][:,1:2]
        p_zz = torch.autograd.grad(p_z, x, torch.ones_like(p_z),
                                   create_graph=True, retain_graph=True)[0][:,2:3]
        lap_p = p_xx + p_yy + p_zz
        res = p_tt - (self.c**2)*lap_p
        return res
    
    def pde_struct_residual(self, x):
        """
        线弹性方程残差： ρₛ u_tt = div(σ)，σ = λ tr(ε) I + 2μ ε
        这里用 t 为第4个坐标计算 u_tt
        """
        x.requires_grad_(True)
        u = self.u_struct(x)  # shape [N,3]
        u_t = torch.autograd.grad(u, x, torch.ones_like(u),
                                  create_graph=True, retain_graph=True)[0][:,3:4]
        u_tt = torch.autograd.grad(u_t, x, torch.ones_like(u_t),
                                   create_graph=True, retain_graph=True)[0][:,3:4]
        # 计算空间梯度（仅取前3列）以构造应变
        grads = []
        for i in range(3):
            u_i = u[:, i:i+1]
            grad_u_i = torch.autograd.grad(u_i, x, torch.ones_like(u_i),
                                           create_graph=True, retain_graph=True)[0][:,0:3]
            grads.append(grad_u_i)
        # 应变: ε_ij = 0.5*(du_i/dx_j + du_j/dx_i)
        eps_xx = grads[0][:,0:1]
        eps_yy = grads[1][:,1:2]
        eps_zz = grads[2][:,2:3]
        eps_xy = 0.5*(grads[0][:,1:2] + grads[1][:,0:1])
        eps_xz = 0.5*(grads[0][:,2:3] + grads[2][:,0:1])
        eps_yz = 0.5*(grads[1][:,2:3] + grads[2][:,1:2])
        trace_eps = eps_xx + eps_yy + eps_zz
        lam = (self.E * self.nu) / ((1+self.nu)*(1-2*self.nu))
        mu = self.E / (2*(1+self.nu))
        # 计算应力 σ = λ trace(ε) I + 2 μ ε
        sigma_xx = lam*trace_eps + 2*mu*eps_xx
        sigma_yy = lam*trace_eps + 2*mu*eps_yy
        sigma_zz = lam*trace_eps + 2*mu*eps_zz
        sigma_xy = 2*mu*eps_xy
        sigma_xz = 2*mu*eps_xz
        sigma_yz = 2*mu*eps_yz
        # 计算 div(σ)
        grad_sigma_xx = torch.autograd.grad(sigma_xx, x, torch.ones_like(sigma_xx),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_xy = torch.autograd.grad(sigma_xy, x, torch.ones_like(sigma_xy),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_xz = torch.autograd.grad(sigma_xz, x, torch.ones_like(sigma_xz),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_yy = torch.autograd.grad(sigma_yy, x, torch.ones_like(sigma_yy),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_yz = torch.autograd.grad(sigma_yz, x, torch.ones_like(sigma_yz),
                                            create_graph=True, retain_graph=True)[0]
        grad_sigma_zz = torch.autograd.grad(sigma_zz, x, torch.ones_like(sigma_zz),
                                            create_graph=True, retain_graph=True)[0]
        div_sigma_x = grad_sigma_xx[:,0:1] + grad_sigma_xy[:,1:2] + grad_sigma_xz[:,2:3]
        div_sigma_y = grad_sigma_xy[:,0:1] + grad_sigma_yy[:,1:2] + grad_sigma_yz[:,2:3]
        div_sigma_z = grad_sigma_xz[:,0:1] + grad_sigma_yz[:,1:2] + grad_sigma_zz[:,2:3]
        res_x = self.rho_s * u_tt - div_sigma_x
        res_y = self.rho_s * u_tt - div_sigma_y
        res_z = self.rho_s * u_tt - div_sigma_z
        residual = torch.cat([res_x, res_y, res_z], dim=1)
        return residual

    def compute_struct_stress(self, x):
        """
        计算结构域在点 x 处的应力张量 σ（shape [N, 3, 3]），
        其中 x 的前3列是空间坐标。
        """
        x.requires_grad_(True)
        u = self.u_struct(x)  # [N,3]
        grads = []
        for i in range(3):
            u_i = u[:, i:i+1]
            grad_u_i = torch.autograd.grad(u_i, x, torch.ones_like(u_i),
                                           create_graph=True, retain_graph=True)[0][:,0:3]
            grads.append(grad_u_i)
        # 构造应变张量 ε
        eps_xx = grads[0][:,0:1]
        eps_yy = grads[1][:,1:2]
        eps_zz = grads[2][:,2:3]
        eps_xy = 0.5*(grads[0][:,1:2] + grads[1][:,0:1])
        eps_xz = 0.5*(grads[0][:,2:3] + grads[2][:,0:1])
        eps_yz = 0.5*(grads[1][:,2:3] + grads[2][:,1:2])
        trace_eps = eps_xx + eps_yy + eps_zz
        lam = (self.E * self.nu) / ((1+self.nu)*(1-2*self.nu))
        mu = self.E / (2*(1+self.nu))
        sigma_xx = lam*trace_eps + 2*mu*eps_xx
        sigma_yy = lam*trace_eps + 2*mu*eps_yy
        sigma_zz = lam*trace_eps + 2*mu*eps_zz
        sigma_xy = 2*mu*eps_xy
        sigma_xz = 2*mu*eps_xz
        sigma_yz = 2*mu*eps_yz
        # 构造应力张量
        sigma = torch.cat([sigma_xx, sigma_xy, sigma_xz,
                           sigma_xy, sigma_yy, sigma_yz,
                           sigma_xz, sigma_yz, sigma_zz], dim=1)
        sigma = sigma.view(-1, 3, 3)
        return sigma

    def interface_residual(self, x_if, normals):
        """
        流固界面残差：
          (i) 应力连续：要求流体侧声压 p 与结构侧法向应力 σ_nn 满足 p + σ_nn = 0
          (ii) 速度连续：流体侧法向速度 = 结构侧法向速度
        normals: [N,3] 单位法向量，指向管壁外侧
        """
        x_if.requires_grad_(True)
        p_val = self.p_fluid(x_if)  # [N,1]
        # 结构应力计算
        sigma = self.compute_struct_stress(x_if)  # [N,3,3]
        # 计算法向分量 σ_nn = nᵀ σ n
        n = normals.view(-1, 3, 1)  # [N,3,1]
        sigma_n = torch.bmm(sigma, n)        # [N,3,1]
        sigma_nn = torch.bmm(n.transpose(1,2), sigma_n)  # [N,1,1]
        sigma_nn = sigma_nn.view(-1,1)
        res_stress = p_val + sigma_nn  # 要求接近0
        
        # 速度连续：计算流体侧法向速度 v_n = (dp/dn)/(rho_f*c)
        p_grad = torch.autograd.grad(p_val, x_if, torch.ones_like(p_val),
                                     create_graph=True, retain_graph=True)[0]
        dp_dn = (p_grad[:,0:3] * normals).sum(dim=1, keepdim=True)
        v_n_fluid = dp_dn / (self.rho_f * self.c)
        # 结构侧法向速度： = d(u·n)/dt
        u_val = self.u_struct(x_if)
        u_n = (u_val * normals).sum(dim=1, keepdim=True)
        u_n_t = torch.autograd.grad(u_n, x_if, torch.ones_like(u_n),
                                    create_graph=True, retain_graph=True)[0][:,3:4]
        v_n_struct = u_n_t
        res_velocity = v_n_fluid - v_n_struct
        
        res = torch.cat([res_stress, res_velocity], dim=1)
        return res
    
    def open_boundary_residual(self, x_open):
        """
        开放边界条件（主管出口 x=x_main_max）： dp/dn + alpha * p = 0, 此处 n=(1,0,0)
        """
        x_open.requires_grad_(True)
        p_val = self.p_fluid(x_open)
        grad_p = torch.autograd.grad(p_val, x_open, torch.ones_like(p_val),
                                     create_graph=True, retain_graph=True)[0]
        dp_dn = grad_p[:,0:1]
        res = dp_dn + alpha_open * p_val
        return res
    
    def data_residual(self, x_data, p_data):
        """
        数据残差：在给定数据点处（例如远端麦克风），预测 p 与测量值比较
        """
        p_pred = self.p_fluid(x_data)
        return p_pred - p_data
    
    def total_loss(self, data):
        """
        组合所有损失项：
         本工程目标：给定声源信号（近端）和远端实际测量，
         前向仿真由 PINN 得到预测的远端信号，
         Loss 仅为 (预测远端信号 - 真实远端信号)²
         但为保证物理正确性，在预训练阶段使用 PDE、界面、开放边界及近端数据损失；
         材料参数优化阶段冻结网络，仅用远端数据 loss。
        data 字典包含：
         'x_f': 流体域 collocation
         'x_s': 结构域 collocation
         'x_if', 'n_if': 流固界面点与法向量
         'x_ob': 开放边界点
         'x_mic_near', 'p_mic_near': 近端声源数据（边界条件）
         'x_mic_far',  'p_mic_far': 远端麦克风数据
        """
        x_f = data['x_f']
        x_s = data['x_s']
        x_if = data['x_if']
        n_if = data['n_if']
        x_ob = data['x_ob']
        x_mic_near = data['x_mic_near']
        p_mic_near = data['p_mic_near']
        x_mic_far = data['x_mic_far']
        p_mic_far = data['p_mic_far']
        
        # 预训练阶段时，综合 PDE、界面、开放边界和近端数据损失
        loss_f = torch.mean(self.pde_fluid_residual(x_f)**2)
        loss_s = torch.mean(self.pde_struct_residual(x_s)**2)
        loss_if = torch.mean(self.interface_residual(x_if, n_if)**2)
        loss_ob = torch.mean(self.open_boundary_residual(x_ob)**2)
        loss_near = torch.mean(self.data_residual(x_mic_near, p_mic_near)**2)
        # 材料参数优化阶段：只使用远端数据 loss
        loss_far = torch.mean(self.data_residual(x_mic_far, p_mic_far)**2)
        
        total = loss_f + loss_s + loss_if + loss_ob + loss_near + loss_far
        loss_dict = {
            'loss_f': loss_f,
            'loss_s': loss_s,
            'loss_if': loss_if,
            'loss_ob': loss_ob,
            'loss_near': loss_near,
            'loss_far': loss_far
        }
        return total, loss_dict

# ===============================
# 预训练阶段：训练网络解决 PDE，接口、边界及近端声源（噪音源）条件
# ===============================
print("Phase 1: 预训练 PINN 解耦合 PDE（固定材料参数）")
N_f_main    = 1000
N_f_branch  = 300
N_s_main    = 1000
N_s_branch  = 300
N_if_main   = 300
N_if_branch = 100
N_ob_main   = 200
N_ob_branch = 50
N_mic_near  = 50
# 流体域：主管内腔及分支内腔
x_f_main = sample_cylinder([x_main_min, x_main_max], 0.0, R_inner_main, N_f_main)
x_f_branch = sample_branch_cylinder(N_f_branch)
x_f = torch.cat([x_f_main, x_f_branch], dim=0)
# 结构域：主管壁和分支壁（环形区域）
x_s_main = sample_annular_cylinder([x_main_min, x_main_max], R_inner_main, R_outer_main, N_s_main)
x_s_branch = sample_branch_annular(N_s_branch)
x_s = torch.cat([x_s_main, x_s_branch], dim=0)
# 流固界面：主管内表面和分支内表面
x_if_main, n_if_main = sample_interface_main(N_if_main)
x_if_branch, n_if_branch = sample_interface_branch(N_if_branch)
x_if = torch.cat([x_if_main, x_if_branch], dim=0)
n_if = torch.cat([n_if_main, n_if_branch], dim=0)
# 开放边界：主管出口和分支出口
x_ob_main = sample_open_boundary_main(N_ob_main)
# 此处仅使用主管出口
x_ob = x_ob_main
# 近端麦克风（声源）数据：位置 near_mic_pos
N_mic_near = 50
def mic_signal_near(t):
    # 声源信号，示例：正弦信号，频率 1000Hz，振幅 1.0 Pa
    return np.sin(2*np.pi*1000*t)
x_mic_near = sample_mic_at(near_mic_pos, N_mic_near)
t_near = x_mic_near[:,3].detach().cpu().numpy().flatten()
p_mic_near_np = mic_signal_near(t_near)
p_mic_near = torch.tensor(p_mic_near_np.reshape(-1,1), dtype=torch.float32).to(device)

# 对于预训练阶段，我们暂时不使用远端数据损失（后续仅用于材料参数调整）
data_pretrain = {
    'x_f': x_f,
    'x_s': x_s,
    'x_if': x_if,
    'n_if': n_if,
    'x_ob': x_ob,
    'x_mic_near': x_mic_near,
    'p_mic_near': p_mic_near,
    # 远端数据留空
    'x_mic_far': x_mic_near,  # 占位，但权重可设为0
    'p_mic_far': p_mic_near
}

# 实例化模型，预训练阶段固定材料参数
fluid_layers = [4, 64, 64, 64, 1]
struct_layers = [4, 64, 64, 64, 3]
model = CoupledPINN(fluid_layers, struct_layers, fixed_material=True).to(device)
optimizer_net = optim.Adam(list(model.fluid_net.parameters())+list(model.struct_net.parameters()), lr=1e-3)
scheduler_net = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_net, mode='min', factor=0.5, patience=500, verbose=True)

nIter_pre = 3000
print("开始预训练...")
for it in range(nIter_pre):
    optimizer_net.zero_grad()
    loss, loss_dict = model.total_loss(data_pretrain)
    loss.backward()
    optimizer_net.step()
    if it % 500 == 0:
        scheduler_net.step(loss)
        print(f"[Pretrain] Iter {it}: Total Loss = {loss.item():.4e}, Loss_far = {loss_dict['loss_far'].item():.4e}")
        print(f"  (仅监控 PDE、界面、开放 BC 和 近端数据)")
print("预训练完成。")

# 冻结网络权重
for param in model.fluid_net.parameters():
    param.requires_grad = False
for param in model.struct_net.parameters():
    param.requires_grad = False
print("网络权重冻结，进入材料参数优化阶段。")

# ===============================
# 材料参数优化阶段：仅用远端数据 loss
# ===============================
# 远端麦克风数据：位于 far_mic_pos
N_mic_far = 50
def mic_signal_far(t):
    # 示例：经过传播衰减和相移，振幅0.5，延迟0.3弧度
    return 0.5 * np.sin(2*np.pi*1000*t - 0.3)
x_mic_far = sample_mic_at(far_mic_pos, N_mic_far)
t_far = x_mic_far[:,3].detach().cpu().numpy().flatten()
p_mic_far_np = mic_signal_far(t_far)
p_mic_far = torch.tensor(p_mic_far_np.reshape(-1,1), dtype=torch.float32).to(device)

data_opt = {
    'x_f': x_f,       # 对于前向仿真，仍需要整个域采样以保证网络内部计算连续性
    'x_s': x_s,
    'x_if': x_if,
    'n_if': n_if,
    'x_ob': x_ob,
    # 近端数据仍作为边界条件（不再作为 loss，可以不加权）
    'x_mic_near': x_mic_near,
    'p_mic_near': p_mic_near,
    # 远端数据作为目标 loss
    'x_mic_far': x_mic_far,
    'p_mic_far': p_mic_far
}

# 在材料参数优化阶段，只解冻材料参数
for param in model.fluid_net.parameters():
    param.requires_grad = False
for param in model.struct_net.parameters():
    param.requires_grad = False

optimizer_mat = optim.Adam([model.log_E, model.nu, model.rho_s_norm], lr=5e-2)
scheduler_mat = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_mat, mode='min', factor=0.5, patience=300, verbose=True)

nIter_opt = 3000
print("开始材料参数优化，仅用远端数据 loss...")
for it in range(nIter_opt):
    optimizer_mat.zero_grad()
    # 此时 total_loss 中的 loss_far 将是唯一起主导作用的项
    loss, loss_dict = model.total_loss(data_opt)
    # 只用远端数据 loss（loss_far）作为优化目标
    loss_far = loss_dict['loss_far']
    loss_far.backward()
    # 保证泊松比在 (0,0.5) 内
    with torch.no_grad():
        model.nu.clamp_(0.0, 0.499)
    optimizer_mat.step()
    if it % 500 == 0:
        scheduler_mat.step(loss_far)
        print(f"[Opt] Iter {it}: Far Mic Loss = {loss_far.item():.4e}")
        print(f"  E = {model.E.item():.3e}, nu = {model.nu.item():.3f}, rho_s = {model.rho_s.item():.2f}")

print("材料参数优化完成。")
print("最终优化结果：")
print(f"  E = {model.E.item():.3e} Pa, nu = {model.nu.item():.3f}, rho_s = {model.rho_s.item():.2f} kg/m³")

# 保存最终模型
torch.save(model.state_dict(), "final_coupled_pinn.pth")
print("模型保存成功。")
