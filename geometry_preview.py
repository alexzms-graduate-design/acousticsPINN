# mesh_preview.py
import meshio
import pyvista as pv
import numpy as np

def convert_meshio_to_pv_by_physical(mesh):
    """
    将 meshio 读取的网格转换为 PyVista 格式(UnstructuredGrid)，
    并根据物理组信息区分不同物理域。
    物理组信息存放在 mesh.cell_data_dict["gmsh:physical"]["tetra"] 中，
    如果不存在则使用默认标签 0。
    返回一个字典，键为物理组 tag，值为相应的 UnstructuredGrid。
    """
    points = mesh.points
    grids = {}
    # 尝试提取 tetra 网格与对应物理组信息
    if "tetra" in mesh.cells_dict:
        tetra_cells = mesh.cells_dict["tetra"]
        # 尝试获取物理组数据
        if "gmsh:physical" in mesh.cell_data_dict and "tetra" in mesh.cell_data_dict["gmsh:physical"]:
            phys_tags = mesh.cell_data_dict["gmsh:physical"]["tetra"]
        else:
            phys_tags = np.zeros(tetra_cells.shape[0], dtype=int)
        print(f"phys_tags: {phys_tags}")
        unique_tags = np.unique(phys_tags)
        for tag in unique_tags:
            # 筛选对应 tag 的单元
            cells_for_tag = tetra_cells[phys_tags == tag]
            n_cells = cells_for_tag.shape[0]
            # VTK tetra 单元格式要求：每个单元数据前有一个数字4表示节点数
            cells_pv = np.hstack([np.full((n_cells, 1), 4, dtype=np.int32), cells_for_tag])
            cells_pv = cells_pv.flatten()
            celltypes = np.full(n_cells, 10, dtype=np.int32)  # VTK tetra 类型为 10
            grid = pv.UnstructuredGrid(cells_pv, celltypes, points)
            grids[tag] = grid
    else:
        # 若无 tetra 单元数据，则返回点云
        grids[0] = pv.PolyData(points)
    return grids

def main():
    msh_file = "y_pipe.msh"
    mesh = meshio.read(msh_file)
    grids = convert_meshio_to_pv_by_physical(mesh)
    print(grids.keys())
    p = pv.Plotter()
    # 按照预设：物理组 2 为管壁，物理组 1 为流体内腔
    if 2 in grids:
        p.add_mesh(grids[2], show_edges=True, color="lightgray", opacity=0.9, label="Pipe Wall")
    else:
        print("物理组 2（管壁）未找到")
    if 1 in grids:
        p.add_mesh(grids[1], show_edges=True, color="skyblue", opacity=0.65, label="Fluid Domain")
    else:
        print("物理组 1（流体内腔）未找到")
    
    # 如有其它物理组，全部显示
    for tag, grid in grids.items():
        if tag not in [1, 2]:
            p.add_mesh(grid, show_edges=True, opacity=0.6, label=f"Group {tag}")

    try:
        p.add_legend()
    except ValueError:
        pass
    p.add_title("Y Pipe Mesh Preview")
    p.show_bounds(grid="front")
    p.show()

if __name__ == "__main__":
    main()
