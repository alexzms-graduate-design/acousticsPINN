# geometry_gmsh.py

import gmsh
import math
import os
import meshio
import numpy as np

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
    angle = 45  # 分支角度
    length_main = 1.5  # 主干总长 1.5m

    # -------------------------------
    # 构建外圆柱体（主管外管与分支外管）
    # -------------------------------
    # 主干外管：沿 x 轴延伸
    cylinder_main_outer = gmsh.model.occ.addCylinder(0, 0, 0, length_main, 0, 0, r_outer)
    # 分支外管：在 x=0.5 处，以45度角向后延伸0.3m
    L_branch = 0.3
    dx = L_branch * math.cos(math.radians(angle))
    dy = L_branch * math.sin(math.radians(angle))
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
    # 将流体内腔定义为物理组 1, 管壁体定义为物理组 2
    # 流体域
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [cylinder_fluid[0][1]], tag=1)
    gmsh.model.setPhysicalName(3, 1, "Fluid Domain")
    
    # 管壁体
    gmsh.model.occ.synchronize()
    gmsh.model.addPhysicalGroup(3, [pipe_wall[0][1]], tag=2)  # 管壁体
    gmsh.model.setPhysicalName(3, 2, "Pipe Wall")

    
    gmsh.model.occ.synchronize()
    gmsh.model.occ.removeAllDuplicates()
    gmsh.model.mesh.generate(3)

    
    msh_file = "y_pipe.msh"
    gmsh.write(msh_file)
    # gmsh.finalize()
    print(f"Generated mesh saved to {msh_file}")

if __name__ == "__main__":
    generate_y_pipe_mesh()
    print("\n--- Running Mesh Check ---")
    try:
        mesh = meshio.read("y_pipe.msh")
        if "tetra" not in mesh.cells_dict:
             print("Error: No tetrahedra found in mesh.")
        else:
            print(f"Total points: {mesh.points.shape[0]}")
            print(f"Tetrahedral cells: {mesh.cells_dict['tetra'].shape[0]}")

            if "gmsh:physical" in mesh.cell_data_dict and "tetra" in mesh.cell_data_dict["gmsh:physical"]:
                 phys_tags = mesh.cell_data_dict['gmsh:physical']['tetra']
                 fluid_count = np.sum(phys_tags == 1)
                 solid_count = np.sum(phys_tags == 2)
                 other_count = len(phys_tags) - fluid_count - solid_count
                 print(f"Fluid cells (tag 1): {fluid_count}")
                 print(f"Solid cells (tag 2): {solid_count}")
                 if other_count > 0: print(f"Other/untagged cells: {other_count}")

                 # Check for shared nodes (heuristic)
                 fluid_nodes = set(mesh.cells_dict["tetra"][phys_tags == 1].flatten())
                 solid_nodes = set(mesh.cells_dict["tetra"][phys_tags == 2].flatten())
                 shared_count = len(fluid_nodes.intersection(solid_nodes))
                 print(f"Nodes unique to fluid: {len(fluid_nodes) - shared_count}")
                 print(f"Nodes unique to solid: {len(solid_nodes) - shared_count}")
                 print(f"Nodes SHARED between fluid/solid: {shared_count}")
                 if shared_count > 0:
                      print("--> Conformal mesh likely generated successfully (shared nodes found).")
                 else:
                      print("--> WARNING: No shared nodes found. Conformal mesh generation might have failed.")

            else:
                 print("Error: Physical tags not found for tetrahedra.")

    except Exception as e:
        print(f"Error reading or checking mesh: {e}")

    finally:
        if gmsh.isInitialized():
             gmsh.finalize()
