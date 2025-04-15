# mesh_preview.py
import meshio
import pyvista as pv
import numpy as np
import os
import sys

def convert_meshio_to_pv(mesh, mesh_format="new"):
    """
    将 meshio 读取的网格转换为 PyVista 格式(UnstructuredGrid)，
    支持新旧两种网格格式。
    
    参数:
        mesh: meshio 读取的网格
        mesh_format: 网格格式类型，"new" 为新的网格格式，"old" 为旧的网格格式
    
    返回:
        grids: 字典，键为物理组 tag，值为相应的 UnstructuredGrid
    """
    points = mesh.points
    grids = {}
    
    # DEBUG: 打印 mesh 对象的结构，帮助诊断问题
    print("\n===== DEBUG: meshio 对象结构 =====")
    print(f"mesh.points.shape: {mesh.points.shape}")
    print(f"mesh 属性: {dir(mesh)}")
    
    if hasattr(mesh, 'cells'):
        print("\n===== cells 属性 =====")
        for i, cell_block in enumerate(mesh.cells):
            print(f"Cell block {i}: type={cell_block.type}, data.shape={cell_block.data.shape}")
    
    if hasattr(mesh, 'cell_data'):
        print("\n===== cell_data 属性 =====")
        for key in mesh.cell_data:
            print(f"Key: {key}")
            for i, data_array in enumerate(mesh.cell_data[key]):
                print(f"  data[{i}].shape={data_array.shape}, unique values={np.unique(data_array)}")
    
    if hasattr(mesh, 'cell_sets'):
        print("\n===== cell_sets 属性 =====")
        print(mesh.cell_sets)
    
    if hasattr(mesh, 'field_data'):
        print("\n===== field_data 属性 =====")
        print(mesh.field_data)
    
    # 标签映射：从标签ID到物理组名称
    tag_names = {}
    if hasattr(mesh, 'field_data'):
        print("\n尝试从 field_data 提取标签名称映射")
        for name, data in mesh.field_data.items():
            if len(data) >= 2:  # field_data 通常存储 [tag, dimension]
                tag = data[0]
                tag_names[tag] = name
                print(f"物理组: {name} -> 标签: {tag}")
    
    # 检查新版 meshio (cells 是列表而非字典)
    if hasattr(mesh, 'cells'):
        # 查找所有四面体单元块及其对应的物理组标签
        tetra_blocks = []
        
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "tetra":
                # 尝试找到对应的物理组标签
                tag = None
                if hasattr(mesh, 'cell_data') and 'gmsh:physical' in mesh.cell_data:
                    if i < len(mesh.cell_data['gmsh:physical']):
                        # 假设每个四面体块在一个物理组内
                        tags = np.unique(mesh.cell_data['gmsh:physical'][i])
                        if len(tags) == 1:  # 确保只有一个标签
                            tag = tags[0]
                            print(f"找到四面体块 {i}，标签为 {tag}")
                            tetra_blocks.append((i, cell_block.data, tag))
                            
                            # 创建该标签的网格
                            cells_for_tag = cell_block.data
                            n_cells = cells_for_tag.shape[0]
                            tag_name = tag_names.get(tag, f"物理组 {tag}")
                            print(f"{tag_name} (标签 {tag}) 的单元数: {n_cells}")
                            
                            # 创建PyVista网格
                            cells_pv = np.hstack([np.full((n_cells, 1), 4, dtype=np.int32), cells_for_tag])
                            cells_pv = cells_pv.flatten()
                            celltypes = np.full(n_cells, 10, dtype=np.int32)  # VTK tetra 类型为 10
                            grid = pv.UnstructuredGrid(cells_pv, celltypes, points)
                            grids[tag] = grid
        
        if tetra_blocks:
            return grids
        else:
            print("警告: 未找到带有物理组标签的四面体单元")
    
    # 兼容旧版 meshio
    elif hasattr(mesh, 'cells_dict') and "tetra" in mesh.cells_dict:
        tetra_cells = mesh.cells_dict["tetra"]
        phys_tags = None
        
        if hasattr(mesh, 'cell_data_dict'):
            if "gmsh:physical" in mesh.cell_data_dict and "tetra" in mesh.cell_data_dict["gmsh:physical"]:
                phys_tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
                
                # 创建网格
                unique_tags = np.unique(phys_tags)
                print(f"\n找到的物理组标签: {unique_tags}")
                
                for tag in unique_tags:
                    tag_name = tag_names.get(tag, f"物理组 {tag}")
                    cells_for_tag = tetra_cells[phys_tags == tag]
                    n_cells = cells_for_tag.shape[0]
                    print(f"{tag_name} (标签 {tag}) 的单元数: {n_cells}")
                    
                    cells_pv = np.hstack([np.full((n_cells, 1), 4, dtype=np.int32), cells_for_tag])
                    cells_pv = cells_pv.flatten()
                    celltypes = np.full(n_cells, 10, dtype=np.int32)
                    grid = pv.UnstructuredGrid(cells_pv, celltypes, points)
                    grids[tag] = grid
                
                return grids
    
    # 如果没有找到有效的网格，返回点云
    if not grids:
        print("警告: 未找到有效的物理组，创建点云")
        grids[0] = pv.PolyData(points)
    
    return grids

def analyze_shared_nodes(mesh):
    """分析流体和固体共享的节点数量，验证网格是否共形"""
    fluid_cells = None
    solid_cells = None
    fluid_tag = None
    solid_tag = None
    
    # 从field_data识别物理组标签
    if hasattr(mesh, 'field_data'):
        for name, data in mesh.field_data.items():
            if len(data) >= 2:
                tag = data[0]
                if name.lower() == 'fluid':
                    fluid_tag = tag
                elif name.lower() == 'wall' or name.lower() == 'solid':
                    solid_tag = tag
    
    if fluid_tag is None or solid_tag is None:
        print(f"警告: 无法确定流体和固体标签，分析中断")
        return
    
    # 查找流体和固体的四面体单元
    if hasattr(mesh, 'cells'):
        for i, cell_block in enumerate(mesh.cells):
            if cell_block.type == "tetra" and hasattr(mesh, 'cell_data') and 'gmsh:physical' in mesh.cell_data:
                if i < len(mesh.cell_data['gmsh:physical']):
                    tags = np.unique(mesh.cell_data['gmsh:physical'][i])
                    if len(tags) == 1:
                        if tags[0] == fluid_tag:
                            fluid_cells = cell_block.data
                        elif tags[0] == solid_tag:
                            solid_cells = cell_block.data
    
    if fluid_cells is None or solid_cells is None:
        print("警告: 未找到流体或固体的四面体单元，无法分析共享节点")
        return
    
    print(f"\n===== 网格共形性分析 =====")
    print(f"识别到流体标签: {fluid_tag}, 固体标签: {solid_tag}")
    
    # 收集流体和固体节点
    fluid_nodes = set()
    solid_nodes = set()
    
    for cell in fluid_cells:
        for node in cell:
            fluid_nodes.add(node)
    
    for cell in solid_cells:
        for node in cell:
            solid_nodes.add(node)
    
    # 计算共享节点
    shared_nodes = fluid_nodes.intersection(solid_nodes)
    
    print(f"流体节点数: {len(fluid_nodes)}")
    print(f"固体节点数: {len(solid_nodes)}")
    print(f"共享节点数: {len(shared_nodes)}")
    
    if len(shared_nodes) > 0:
        print("✓ 网格共形检验通过！在流体-固体界面上存在共享节点。")
    else:
        print("✗ 网格共形检验失败！未在流体-固体界面上找到共享节点。")
    
    return fluid_tag, solid_tag, list(shared_nodes)

def main():
    # 支持命令行参数指定网格文件
    if len(sys.argv) > 1:
        msh_file = sys.argv[1]
    else:
        # 默认尝试新生成的网格文件
        msh_file = "y_pipe_conformal.msh"
        # 如果文件不存在，尝试其他可能的文件名
        if not os.path.exists(msh_file):
            alternative_files = ["y_pipe_simple.msh", "y_pipe_compound.msh", "y_pipe.msh", "y_pipe_conformal_boolean.msh"]
            for alt_file in alternative_files:
                if os.path.exists(alt_file):
                    msh_file = alt_file
                    break
    
    print(f"预览网格文件: {msh_file}")
    mesh = meshio.read(msh_file)
    
    # 分析共享节点并获取流体和固体标签
    result = analyze_shared_nodes(mesh)
    shared_info = None
    if result:
        fluid_tag, solid_tag, shared_nodes = result
        shared_info = (fluid_tag, solid_tag, shared_nodes)
    
    # 转换为 PyVista 网格
    grids = convert_meshio_to_pv(mesh)
    print(f"物理组标签: {list(grids.keys())}")
    
    # 创建 PyVista 可视化
    p = pv.Plotter()
    
    # 获取物理组名称映射
    tag_names = {}
    if hasattr(mesh, 'field_data'):
        for name, data in mesh.field_data.items():
            if len(data) >= 2:
                tag = data[0]
                tag_names[tag] = name
    
    # 显示所有物理组网格
    for tag, grid in grids.items():
        if tag == 0:
            # 默认的全部网格
            p.add_mesh(grid, show_edges=True, color="gray", label="完整网格")
            continue
            
        # 确定合适的颜色和标签
        group_name = tag_names.get(tag, f"物理组 {tag}")
        if 'fluid' in group_name.lower():
            p.add_mesh(grid, show_edges=True, color="skyblue", opacity=0.65, label=group_name)
        elif 'wall' in group_name.lower() or 'solid' in group_name.lower():
            p.add_mesh(grid, show_edges=True, color="lightgray", opacity=0.9, label=group_name)
        elif 'interface' in group_name.lower():
            p.add_mesh(grid, show_edges=True, color="green", opacity=0.8, label=group_name)
        else:
            p.add_mesh(grid, show_edges=True, opacity=0.6, label=group_name)
    
    # 尝试显示共享节点（流体-固体界面）
    if shared_info:
        try:
            fluid_tag, solid_tag, shared_nodes = shared_info
            if len(shared_nodes) > 0:
                print(f"\n找到 {len(shared_nodes)} 个流体-固体共享节点，将高亮显示")
                
                # 创建共享节点的点云
                shared_coords = mesh.points[shared_nodes]
                shared_cloud = pv.PolyData(shared_coords)
                
                # 添加共享节点点云至场景（醒目颜色且加大点尺寸）
                p.add_mesh(shared_cloud, color="red", point_size=8, 
                           render_points_as_spheres=True, label="流体-固体界面共享节点")
                
                # 添加文本说明共享节点数量
                p.add_text(f"共享节点数: {len(shared_nodes)}", position="upper_right", 
                           font_size=14, color="red")
        except Exception as e:
            print(f"显示共享节点时出错: {e}")
    
    # 添加图例和标题
    try:
        p.add_legend()
    except ValueError:
        pass
    
    p.add_title(f"Y型管道网格预览: {os.path.basename(msh_file)}")
    p.show_bounds(grid="front")
    p.show()

if __name__ == "__main__":
    main()
