# geometry_gmsh.py

import gmsh
import math
import os

# 自定义网格尺寸参数（单位：米）
mesh_size = 0.02  # 调整此参数以改变网格精度

def generate_y_pipe_mesh():
    gmsh.initialize()
    gmsh.model.add("Y_Pipe")
    # # 设置全局网格尺寸
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)

    # 参数定义
    r_outer = 0.05   # 外半径 5cm
    r_inner = 0.045  # 内半径 4.5cm
    length_main = 1.5  # 主干总长 1.5m

    # -------------------------------
    # 构建外圆柱体（主管外管与分支外管）
    # -------------------------------
    # 主干外管：沿 x 轴延伸
    cylinder_main_outer = gmsh.model.occ.addCylinder(0, 0, 0, length_main, 0, 0, r_outer)
    # 分支外管：在 x=0.5 处，以45度角向后延伸0.3m
    L_branch = 0.3
    dx = L_branch * math.cos(math.radians(45))
    dy = L_branch * math.sin(math.radians(45))
    cylinder_branch_outer = gmsh.model.occ.addCylinder(0.5, 0, 0, -dx, dy, 0, r_outer)
    
    gmsh.model.occ.synchronize()
    # 外管并集：使用 fuse 将主管外管与分支外管合并
    outer_union, _ = gmsh.model.occ.fuse([(3, cylinder_main_outer)], [(3, cylinder_branch_outer)])
    
    # -------------------------------
    # 构建内圆柱体（主管内管与分支内管）
    # -------------------------------
    cylinder_main_inner = gmsh.model.occ.addCylinder(0, 0, 0, length_main, 0, 0, r_inner)
    cylinder_branch_inner = gmsh.model.occ.addCylinder(0.5, 0, 0, -dx, dy, 0, r_inner)
    
    gmsh.model.occ.synchronize()
    inner_union, _ = gmsh.model.occ.fuse([(3, cylinder_main_inner)], [(3, cylinder_branch_inner)])
    
    # -------------------------------
    # 构建管壁体：外管并集减去内管并集
    # -------------------------------
    gmsh.model.occ.synchronize()
    pipe_wall, _ = gmsh.model.occ.cut(outer_union, inner_union)
    
    # 再次构建流体域
    gmsh.model.occ.synchronize()
    cylinder_fluid1 = gmsh.model.occ.addCylinder(0, 0, 0, length_main, 0, 0, r_inner)
    cylinder_fluid2 = gmsh.model.occ.addCylinder(0.5, 0, 0, -dx, dy, 0, r_inner)
    cylinder_fluid, _ = gmsh.model.occ.fuse([(3, cylinder_fluid1)], [(3, cylinder_fluid2)])

    # 定义物理组
    # 将管壁体定义为物理组 1，流体内腔（内并集）定义为物理组 2
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [pipe_wall[0][1]], tag=2)  # 管壁体
    gmsh.model.setPhysicalName(3, 2, "Pipe Wall")
    gmsh.model.occ.synchronize()
    # 流体域
    gmsh.model.addPhysicalGroup(3, [cylinder_fluid[0][1]], tag=1)
    gmsh.model.setPhysicalName(3, 1, "Fluid Domain")
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(3)

    
    msh_file = "y_pipe.msh"
    gmsh.write(msh_file)
    # gmsh.finalize()
    print(f"Generated mesh saved to {msh_file}")

if __name__ == "__main__":
    generate_y_pipe_mesh()
    import meshio
    mesh = meshio.read("y_pipe.msh")
    print(mesh.cell_data_dict.keys())
    print(f"cell_data_dict['gmsh:physical']['tetra'].shape: {mesh.cell_data_dict['gmsh:physical']['tetra'].shape}")
    print(f"tetra shape(cells[0].data): {mesh.cells[0].data.shape}")
    print(f"tetra shape(cells[1].data): {mesh.cells[1].data.shape}")
    print(f"# of physics group=1: {len(mesh.cell_data_dict['gmsh:physical']['tetra'][mesh.cell_data_dict['gmsh:physical']['tetra'] == 1])}")
    print(f"# of physics group=2: {len(mesh.cell_data_dict['gmsh:physical']['tetra'][mesh.cell_data_dict['gmsh:physical']['tetra'] == 2])}")
