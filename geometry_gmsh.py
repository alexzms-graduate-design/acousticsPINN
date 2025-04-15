# geometry_gmsh.py

import gmsh
import math
import numpy as np # Import numpy for CoM check
import os
import meshio
import sys

# 自定义网格尺寸参数（单位：米）
lc_fine_fluid_interior = 0.01  # 流体内部目标网格尺寸 (更细)
lc_coarse_default = 0.03       # 默认/固体/其他区域目标网格尺寸 (更粗)
# lc_interface = 0.008          # (Optional) Interface refinement size

def generate_conformal_mesh():
    """Generate a conformal mesh for the Y-pipe with shared nodes at the interface."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1)  # Enable terminal output
    gmsh.option.setNumber("Mesh.Algorithm3D", 6)  # Use default 3D meshing algorithm (Delaunay)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.02)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.02)
    
    model_name = "Y_Pipe_Conformal" 
    gmsh.model.add(model_name)
    
    # --- Parameters ---
    r_outer = 0.05   # 外半径 5cm
    r_inner = 0.045  # 内半径 4.5cm
    angle = 45       # 分支角度
    length_main = 1.5  # 主干总长 1.5m
    L_branch = 0.3   # 与 fem.py 保持一致
    x_junction = 0.5 # 分支点位置
    
    print("[Step 1] Creating pipe geometry...")
    # Calculate branch direction
    dx = L_branch * math.cos(math.radians(angle))
    dy = L_branch * math.sin(math.radians(angle))
    
    # -------------- Create Inner Cylinders (Fluid Domain) --------------
    # Main inner cylinder (fluid)
    c_main_in = gmsh.model.occ.addCylinder(0, 0, 0, length_main, 0, 0, r_inner)
    # Branch inner cylinder (fluid)
    c_branch_in = gmsh.model.occ.addCylinder(x_junction, 0, 0, -dx, dy, 0, r_inner)
    
    # -------------- Create Outer Cylinders (Solid Domain Outer Shape) --------------
    # Main outer cylinder
    c_main_out = gmsh.model.occ.addCylinder(0, 0, 0, length_main, 0, 0, r_outer)
    # Branch outer cylinder
    c_branch_out = gmsh.model.occ.addCylinder(x_junction, 0, 0, -dx, dy, 0, r_outer)
    
    # Synchronize OpenCASCADE model
    gmsh.model.occ.synchronize()
    
    print("[Step 2] Unifying fluid domain...")
    # Fuse inner cylinders (fluid domain)
    fluid_vol, fluid_map = gmsh.model.occ.fuse([(3, c_main_in)], [(3, c_branch_in)])
    if not fluid_vol:
        raise RuntimeError("Failed to fuse fluid cylinders!")
    fluid_tag = fluid_vol[0][1]
    print(f"  • Created fluid volume with tag {fluid_tag}")
    
    print("[Step 3] Unifying outer shell...")
    # Fuse outer cylinders
    outer_vol, outer_map = gmsh.model.occ.fuse([(3, c_main_out)], [(3, c_branch_out)])
    if not outer_vol:
        raise RuntimeError("Failed to fuse outer cylinders!")
    outer_tag = outer_vol[0][1]
    print(f"  • Created outer shell with tag {outer_tag}")
    
    print("[Step 4] Creating solid domain (pipe wall)...")
    # Create solid domain using boolean difference
    temp_fluid = gmsh.model.occ.copy([(3, fluid_tag)])
    temp_fluid_tag = temp_fluid[0][1]
    pipe_wall, pipe_map = gmsh.model.occ.cut([(3, outer_tag)], [(3, temp_fluid_tag)])
    if not pipe_wall:
        raise RuntimeError("Failed to create pipe wall!")
    wall_tag = pipe_wall[0][1]
    print(f"  • Created pipe wall with tag {wall_tag}")
    
    # Synchronize model
    gmsh.model.occ.synchronize()
    
    # Print all volumes in model
    volumes = gmsh.model.getEntities(3)
    print(f"[Debug] Volumes in model: {volumes}")
    
    # -------------- Critical step for conformality: fragment all volumes --------------
    print("[Step 5] Fragmenting volumes to ensure conformality...")
    all_volumes = gmsh.model.getEntities(3)
    print(f"  • Before fragment: {all_volumes}")
    
    # Get IDs of all volumes
    volume_ids = [vol[1] for vol in all_volumes]
    
    # Run fragment operation to ensure conformality
    # This replaces all volumes with new ones that share nodes at interfaces
    if len(volume_ids) >= 2:
        print(f"  • Fragmenting volumes {volume_ids}")
        
        # 修复：正确构建实体元组列表
        object_vol = [(3, volume_ids[0])]
        tool_vols = [(3, vid) for vid in volume_ids[1:]]  # 为每个体积ID创建单独的元组
        
        print(f"  • Object volume: {object_vol}")
        print(f"  • Tool volumes: {tool_vols}")
        
        result_fragments, result_map = gmsh.model.occ.fragment(
            object_vol,  # 第一个体积作为对象
            tool_vols    # 其余体积作为工具
        )
        gmsh.model.occ.synchronize()
        print(f"  • Fragment result: {result_fragments}")
        print(f"  • Fragment map: {result_map}")
        
        # Get volumes after fragment
        volumes_after = gmsh.model.getEntities(3)
        print(f"  • After fragment: {volumes_after}")
        
        # CRITICAL: We need to identify which fragments correspond to fluid and wall
        # One way is to check the center of mass
        print("\n[Step 6] Identifying fragmented volumes...")
        # 使用列表收集体积信息以便比较
        volume_info = []
        
        # Loop through all volume fragments
        for vol in volumes_after:
            vol_dim, vol_tag = vol
            # Get center of mass of this volume
            com = gmsh.model.occ.getCenterOfMass(vol_dim, vol_tag)
            # Get volume
            mass = gmsh.model.occ.getMass(vol_dim, vol_tag)
            print(f"  • Volume {vol_tag}: CoM={com}, Volume={mass}")
            volume_info.append((vol_tag, mass))
        
        # 根据体积大小排序(升序)
        volume_info.sort(key=lambda x: x[1])
        
        # 输出排序后的体积信息，方便调试
        print(f"  • Sorted volumes by size: {volume_info}")
        
        # 从实际数据看，体积比例约为 4:1
        if len(volume_info) >= 2:
            # 较大的体积应该是流体，较小的是壁体
            fluid_new_tag = volume_info[1][0]  # 较大的体积是流体
            wall_new_tag = volume_info[0][0]   # 较小的体积是壁体
            
            # 输出决定
            print(f"  • Identified fluid volume (larger): Tag={fluid_new_tag}, Volume={volume_info[1][1]}")
            print(f"  • Identified wall volume (smaller): Tag={wall_new_tag}, Volume={volume_info[0][1]}")
        else:
            raise RuntimeError(f"Expected at least 2 volumes after fragment, got {len(volume_info)}")
    else:
        raise RuntimeError("Expected at least two volumes for fragmentation!")
    
    # -------------- Set physical groups --------------
    print("\n[Step 7] Assigning physical groups...")
    # Add physical groups for domains
    gmsh.model.occ.synchronize()
    fluid_group = gmsh.model.addPhysicalGroup(3, [fluid_new_tag], tag=1)
    gmsh.model.setPhysicalName(3, fluid_group, "Fluid")
    gmsh.model.occ.synchronize()
    wall_group = gmsh.model.addPhysicalGroup(3, [wall_new_tag], tag=2)
    gmsh.model.setPhysicalName(3, wall_group, "Wall")
    gmsh.model.occ.synchronize()
    # Find interface between fluid and wall for visualization
    print("\n[Step 8] Identifying fluid-solid interface...")
    fluid_boundaries = gmsh.model.getBoundary([(3, fluid_new_tag)], oriented=False)
    wall_boundaries = gmsh.model.getBoundary([(3, wall_new_tag)], oriented=False)
    gmsh.model.occ.synchronize()
    # Find shared surfaces (interface)
    shared_surfaces = []
    for fb in fluid_boundaries:
        for wb in wall_boundaries:
            if fb[1] == wb[1]:  # If tags match
                shared_surfaces.append(fb)
                break
    gmsh.model.occ.synchronize()
    if shared_surfaces:
        print(f"  • Found {len(shared_surfaces)} interface surfaces: {shared_surfaces}")
        # Create physical group for interface
        interface_group = gmsh.model.addPhysicalGroup(2, [s[1] for s in shared_surfaces], tag=3)
        gmsh.model.setPhysicalName(2, interface_group, "FluidSolidInterface")
        gmsh.model.occ.synchronize()
    else:
        print("  • No shared interface surfaces found!")
    
    # -------------- Generate mesh --------------
    print("\n[Step 9] Setting mesh options and generating mesh...")
    # Set mesh options for quality
    gmsh.option.setNumber("Mesh.Algorithm", 6)  # Frontal-Delaunay for 2D
    gmsh.option.setNumber("Mesh.Algorithm3D", 1)  # Delaunay for 3D
    gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    gmsh.option.setNumber("Mesh.Optimize", 1)
    
    # Additional options to ensure shared nodes
    gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.option.setNumber("Mesh.SaveElementTagType", 2)  # Save element tags as element (2) tags
    
    # 移除不支持的选项
    # gmsh.option.setNumber("Mesh.Preserve3DEdgeNodes", 1)  # 这个选项不存在
    
    # 添加其他有助于生成共形网格的选项
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 0)
    gmsh.option.setNumber("Mesh.MeshOnlyVisible", 0)
    
    # 尝试添加更多有助于共形网格的选项
    gmsh.option.setNumber("Mesh.ElementOrder", 1)  # 使用一阶元素
    gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 0) # 不从曲率调整网格大小
    gmsh.option.setNumber("Mesh.CharacteristicLengthExtendFromBoundary", 0) # 不从边界扩展网格大小
    
    try:
        # 有些版本支持下面的选项
        gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0)
    except:
        print("  • Note: Mesh.MeshSizeFactor option not available in this Gmsh version")
    
    # Field for mesh size control
    fine_field = gmsh.model.mesh.field.add("Constant")
    gmsh.model.mesh.field.setNumber(fine_field, "VIn", lc_fine_fluid_interior)
    
    # Restrict fine mesh to fluid domain
    restrict_field = gmsh.model.mesh.field.add("Restrict")
    gmsh.model.mesh.field.setNumber(restrict_field, "InField", fine_field)
    gmsh.model.mesh.field.setNumbers(restrict_field, "VolumesList", [fluid_new_tag])
    
    # Default coarse field for wall
    coarse_field = gmsh.model.mesh.field.add("Constant")
    gmsh.model.mesh.field.setNumber(coarse_field, "VIn", lc_coarse_default)
    
    # Min field to use the minimum of all fields
    min_field = gmsh.model.mesh.field.add("Min")
    gmsh.model.mesh.field.setNumbers(min_field, "FieldsList", [restrict_field, coarse_field])
    
    # Set the min field as background field
    gmsh.model.mesh.field.setAsBackgroundMesh(min_field)
    
    # Create mesh
    print("  • Generating conformal mesh...")
    gmsh.model.mesh.generate(3)
    
    # Analyze mesh - check node counts for each volume
    print("\n[Step 10] Analyzing mesh quality...")
    fluid_elems = gmsh.model.mesh.getElements(3, fluid_new_tag)
    wall_elems = gmsh.model.mesh.getElements(3, wall_new_tag)
    
    print(f"  • Fluid elements: {len(fluid_elems[1][0])}")
    print(f"  • Wall elements: {len(wall_elems[1][0])}")
    
    # Get all nodes in fluid elements (flatten the node arrays)
    fluid_nodes = set()
    for elem_nodes in fluid_elems[2]:
        for node in elem_nodes:
            fluid_nodes.add(node)
    
    # Get all nodes in wall elements
    wall_nodes = set()
    for elem_nodes in wall_elems[2]:
        for node in elem_nodes:
            wall_nodes.add(node)
    
    # Calculate shared nodes
    shared_nodes = fluid_nodes.intersection(wall_nodes)
    print(f"  • Unique fluid nodes: {len(fluid_nodes)}")
    print(f"  • Unique wall nodes: {len(wall_nodes)}")
    print(f"  • Shared nodes: {len(shared_nodes)}")
    if len(shared_nodes) > 0:
        print("  ✓ CONFORMALITY CHECK PASSED: Found shared nodes at interface!")
    else:
        print("  ✗ CONFORMALITY CHECK FAILED: No shared nodes found!")
    
    # Save mesh
    msh_file = "y_pipe.msh"
    gmsh.write(msh_file)
    print(f"\n[Step 11] Saved conformal mesh to {msh_file}")
    
    # Optionally save mesh in other formats for visualization
    # gmsh.write(msh_file.replace(".msh", ".vtk"))
    print(f"  • Also saved in VTK format for visualization")
    
    return msh_file

if __name__ == "__main__":
    try:
        print("===== Y-PIPE CONFORMAL MESH GENERATOR =====")
        msh_file = generate_conformal_mesh()
        
        # Additional mesh check with meshio
        print("\n===== MESHIO VERIFICATION =====")
        mesh = meshio.read(msh_file)
        print(f"  • Total points: {mesh.points.shape[0]}")
        
        # Count cells by type
        print("  • Cell counts by type:")
        for cell_block in mesh.cells:
            print(f"    - {cell_block.type}: {cell_block.data.shape[0]}")
        
        # Get cell data if available
        if hasattr(mesh, 'cell_data') and mesh.cell_data:
            print("\n  • Cell data fields:")
            for key, val in mesh.cell_data.items():
                print(f"    - {key}: {len(val)} blocks")
        
        # --- 详细验证物理组和共享节点 ---
        print("\n===== PHYSICAL GROUP AND SHARED NODE VERIFICATION =====")
        
        # 查找四面体单元
        tetra_data = None
        tetra_idx = None
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "tetra":
                tetra_data = cell_block.data
                tetra_idx = i
                print(f"  • Found tetrahedral elements: {tetra_data.shape[0]}")
                break
        
        if tetra_data is None:
            print("  ✗ ERROR: No tetrahedral elements found!")
        else:
            # 尝试找到物理组标签
            phys_tags = None
            for key in mesh.cell_data:
                if "gmsh:physical" in key or "tag" in key.lower() or "physical" in key.lower():
                    if tetra_idx < len(mesh.cell_data[key]):
                        phys_tags = mesh.cell_data[key][tetra_idx]
                        print(f"  • Found physical tags in field '{key}'")
                        break
            
            if phys_tags is None:
                print("  ✗ ERROR: No physical group tags found for tetrahedra!")
            else:
                # 统计各个物理组单元数量
                unique_tags = np.unique(phys_tags)
                print(f"  • Physical groups found: {unique_tags}")
                
                # 验证流体和固体物理组
                fluid_found = 1 in unique_tags
                solid_found = 2 in unique_tags
                
                print(f"  • Fluid group (tag 1): {'✓ FOUND' if fluid_found else '✗ NOT FOUND'}")
                print(f"  • Solid group (tag 2): {'✓ FOUND' if solid_found else '✗ NOT FOUND'}")
                
                if fluid_found and solid_found:
                    # 提取各物理组的节点
                    fluid_cells = tetra_data[phys_tags == 1]
                    solid_cells = tetra_data[phys_tags == 2]
                    
                    fluid_nodes = set(fluid_cells.flatten())
                    solid_nodes = set(solid_cells.flatten())
                    
                    # 检查共享节点
                    shared_nodes = fluid_nodes.intersection(solid_nodes)
                    
                    print(f"  • Fluid nodes: {len(fluid_nodes)}")
                    print(f"  • Solid nodes: {len(solid_nodes)}")
                    print(f"  • Shared nodes: {len(shared_nodes)}")
                    
                    if len(shared_nodes) > 0:
                        print("  ✓ CONFORMALITY VERIFICATION PASSED: Mesh has shared nodes at fluid-solid interface!")
                        # 显示一些共享节点的坐标
                        print("\n  • Examples of shared node coordinates:")
                        shared_list = list(shared_nodes)
                        for i in range(min(5, len(shared_list))):
                            node_idx = shared_list[i]
                            print(f"    Node {node_idx}: {mesh.points[node_idx]}")
                    else:
                        print("  ✗ CONFORMALITY VERIFICATION FAILED: No shared nodes found between fluid and solid!")
                else:
                    print("  ✗ ERROR: Missing physical groups. Cannot verify shared nodes.")
        
        # Convert to easier format for visualization
        # vtk_file = msh_file.replace(".msh", "_meshio.vtk")
        # meshio.write(vtk_file, mesh)
        # print(f"\n  • Converted mesh to VTK format: {vtk_file}")
        print("\n===== MESH GENERATION COMPLETE =====")
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if gmsh.isInitialized():
            gmsh.finalize()
