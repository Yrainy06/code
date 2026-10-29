import os
import numpy as np
import h5py
import pyvista as pv
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
from pathlib import Path


def parse_inp_file(inp_file):
    """Parse an Abaqus .inp mesh file to extract nodes and elements (with type & ELSET)."""
    nodes = {}
    elements = []  # each item: {"type": str, "elset": str, "conn": [int,...]}

    with open(inp_file, 'r') as f:
        lines = f.readlines()

    # --- Parse *NODE block(s)
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if line.upper().startswith('*NODE'):
            i += 1
            # Read until next section (line starting with '*') or EOF
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith('*'):
                parts = [p.strip() for p in lines[i].split(',')]
                if len(parts) >= 4:
                    node_id = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    nodes[node_id] = [x, y, z]
                i += 1
            # continue without increment; main loop will handle next line
        i += 1

    # --- Parse *ELEMENT blocks (can appear multiple times with different types/ELSETs)
    i = 0
    while i < n:
        line = lines[i].strip()
        if line.upper().startswith('*ELEMENT'):
            # Example: *ELEMENT, type=T3D2, ELSET=Line1
            header = line.upper()
            etype = None
            elset = None
            # extract type and ELSET (robust to order/whitespace)
            for token in header.split(','):
                token = token.strip()
                if token.startswith('TYPE='):
                    etype = token.split('=', 1)[1].strip()
                elif token.startswith('ELSET='):
                    elset = token.split('=', 1)[1].strip()

            i += 1
            # read element connectivity until next '*' or EOF or blank line
            while i < n and lines[i].strip() and not lines[i].lstrip().startswith('*'):
                # allow comments or decorative lines to be skipped
                raw = lines[i].strip()
                if raw.startswith('**') or raw.startswith('*******'):
                    i += 1
                    continue
                parts = [p.strip() for p in raw.split(',') if p.strip() != '']
                # Expect: elem_id, n1, n2, ...
                if len(parts) >= 2:
                    # skip element id
                    try:
                        _ = int(parts[0])
                        conn = [int(p) for p in parts[1:]]
                        elements.append({"type": etype, "elset": elset, "conn": conn})
                    except ValueError:
                        # non-data line; skip
                        pass
                i += 1
            continue  # re-evaluate current i (already advanced)
        i += 1

    return nodes, elements

def _infer_vtk_cell_type(etype, conn_len):
    """
    Map Abaqus element type to VTK cell type.
    Supports common 2D: T3D2 (line), CPS3/CPE3 (tri), CPS4/CPE4 (quad).
    """
    if etype is None:
        # fallback on node count
        if conn_len == 2:
            return pv.CellType.LINE
        if conn_len == 3:
            return pv.CellType.TRIANGLE
        if conn_len == 4:
            return pv.CellType.QUAD
    et = etype.upper()
    if et.startswith('T3D2') or et == 'B31' or et == 'T2D2':
        return pv.CellType.LINE
    if et.startswith('CPS3') or et.startswith('CPE3') or 'TRI' in et:
        return pv.CellType.TRIANGLE
    if et.startswith('CPS4') or et.startswith('CPE4') or 'QUAD' in et:
        return pv.CellType.QUAD
    # default by count
    if conn_len == 2:
        return pv.CellType.LINE
    if conn_len == 3:
        return pv.CellType.TRIANGLE
    if conn_len == 4:
        return pv.CellType.QUAD
    raise ValueError(f"Unsupported element type {etype} with {conn_len} nodes.")

def create_pyvista_mesh(nodes, elements):
    """Create a PyVista mixed UnstructuredGrid from nodes and element dicts."""
    # Map node IDs to 0-based indices and build points array
    node_ids = sorted(nodes.keys())
    id2idx = {nid: i for i, nid in enumerate(node_ids)}
    points = np.array([nodes[nid] for nid in node_ids], dtype=float)
    num_nodes = len(node_ids)

    # Build VTK 'cells' array and 'celltypes' for a mixed grid
    cells = []
    celltypes = []
    is_boundary = []  # 1 for line (boundaries), 0 otherwise
    elset_ids = []    # numeric id per cell; helpful for later selection
    elset_to_int = {}
    next_elset_int = 1

    for elem in elements:
        conn_ids = elem["conn"]
        etype = elem.get("type", None)
        elset = elem.get("elset", None)

        # Map original node ids to 0-based indices
        try:
            conn_idx = [id2idx[nid] for nid in conn_ids]
        except KeyError as e:
            raise KeyError(f"Element references unknown node id {e}.")

        vtk_type = _infer_vtk_cell_type(etype, len(conn_idx))
        cells.extend([len(conn_idx)] + conn_idx)
        celltypes.append(vtk_type)
        is_boundary.append(1 if vtk_type == pv.CellType.LINE else 0)

        if elset is None:
            elset = "UNSPECIFIED"
        if elset not in elset_to_int:
            elset_to_int[elset] = next_elset_int
            next_elset_int += 1
        elset_ids.append(elset_to_int[elset])

    if len(cells) == 0:
        # No elements parsed — fall back to a point cloud PolyData (rare)
        mesh = pv.PolyData(points)
        return mesh, num_nodes

    cells = np.array(cells, dtype=np.int64)
    celltypes = np.array(celltypes, dtype=np.uint8)

    # Create mixed unstructured grid
    mesh = pv.UnstructuredGrid(cells, celltypes, points)

    # Attach minimal, useful metadata
    mesh.cell_data["is_boundary"] = np.array(is_boundary, dtype=np.int8)
    mesh.cell_data["elset_id"] = np.array(elset_ids, dtype=np.int32)

    # Store a lookup table for elset names
    # (PyVista field_data is numeric; we store a compact mapping table)
    # Two-column "codes": id, string_index. For convenience, also store the names list.
    # If you don't need this, you can remove the next two lines.
    mesh.field_data["ELSET_NAMES"] = np.array([hash(name) for name in elset_to_int.keys()], dtype=np.int64)
    mesh.field_data["ELSET_IDS"] = np.array(list(elset_to_int.values()), dtype=np.int32)

    return mesh, num_nodes


def plot_comparison(mesh, predicted_data, target_data, var_names, timestep, save_path, trajectory_num):
    """Create comparison plots for predicted vs target at a specific timestep."""
    # Set up the plotter with 2 rows (predicted, target) and 4 columns (variables)
    plotter = pv.Plotter(shape=(2, 4), window_size=(3200, 1600), off_screen=True)
    
    # Variable names for 2D Euler equations
    if var_names is None:
        var_names = ['Density', 'Velocity X', 'Velocity Y', 'Pressure']
    
    # Use ONLY target_data for global min/max (color scaling)
    vmin = []
    vmax = []
    for i in range(4):
        vmin.append(np.min(target_data[:, i]))
        vmax.append(np.max(target_data[:, i]))
    
    # Plot each variable
    for col in range(4):
        # Predicted (top row)
        plotter.subplot(0, col)
        mesh_copy = mesh.copy()
        mesh_copy.point_data[var_names[col]] = predicted_data[:, col]
        plotter.add_mesh(mesh_copy, scalars=var_names[col], 
                        cmap='jet', clim=[vmin[col], vmax[col]],
                        show_edges=False, lighting=False)
        plotter.add_text(f'Predicted - {var_names[col]}', font_size=14)
        plotter.view_xy()  # Top-down view for 2D
        plotter.camera.zoom(1.1)
        
        # Target (bottom row)
        plotter.subplot(1, col)
        mesh_copy = mesh.copy()
        mesh_copy.point_data[var_names[col]] = target_data[:, col]
        plotter.add_mesh(mesh_copy, scalars=var_names[col], 
                        cmap='jet', clim=[vmin[col], vmax[col]],
                        show_edges=False, lighting=False)
        plotter.add_text(f'Target - {var_names[col]}', font_size=14)
        plotter.view_xy()  # Top-down view for 2D
        plotter.camera.zoom(1.1)
    
    # Add title
    title = f'Trajectory {trajectory_num} - Timestep {timestep}'
    plotter.add_text(title, position='lower_edge', font_size=20)
    
    # Save the plot
    filename = f'comparison_timestep_{timestep:04d}.png'
    filepath = os.path.join(save_path, filename)
    plotter.screenshot(filepath)
    plotter.close()
    
    print(f"Saved: {filepath}")

def plot_error_maps(mesh, predicted_data, target_data, var_names, timestep, save_path, trajectory_num):
    """Create error maps showing the difference between predicted and target."""
    plotter = pv.Plotter(shape=(1, 4), window_size=(3200, 800), off_screen=True)
    
    if var_names is None:
        var_names = ['Density', 'Velocity X', 'Velocity Y', 'Pressure']
    
    # Plot error for each variable
    for col in range(4):
        plotter.subplot(0, col)
        mesh_copy = mesh.copy()
        
        # Calculate absolute error
        error = np.abs(predicted_data[:, col] - target_data[:, col])
        mesh_copy.point_data['Error'] = error
        
        plotter.add_mesh(mesh_copy, scalars='Error', 
                        cmap='hot_r', 
                        show_edges=False, lighting=False)
        plotter.add_text(f'|Error| - {var_names[col]}', font_size=14)
        plotter.view_xy()  # Top-down view for 2D
        plotter.camera.zoom(1.1)
    
    # Add title
    title = f'Trajectory {trajectory_num} - Error Map - Timestep {timestep}'
    plotter.add_text(title, position='upper_edge', font_size=20)
    
    # Save the plot
    filename = f'error_map_timestep_{timestep:04d}.png'
    filepath = os.path.join(save_path, filename)
    plotter.screenshot(filepath)
    plotter.close()
    
    print(f"Saved error map: {filepath}")


def main():
    # Configuration
    timestep_interval = 6  # Plot every  timesteps
    var_names = ['rho', 'v1', 'v2', 'pres']
    
    # Create compare directory
    compare_dir = 'compare'
    os.makedirs(compare_dir, exist_ok=True)
    
    # Get list of trajectory numbers
    dataset_dir = 'Dataset'
    result_dir = 'result'
    
    # Find all trajectory folders/files
    trajectory_nums = []
    for item in os.listdir(dataset_dir):
        if os.path.isdir(os.path.join(dataset_dir, item)) and item.isdigit():
            trajectory_nums.append(int(item))
    
    trajectory_nums.sort()
    
    print(f"Found {len(trajectory_nums)} trajectories: {trajectory_nums}")
    
    # Process each trajectory
    for traj_num in trajectory_nums:
        print(f"\nProcessing trajectory {traj_num}...")
        
        # Create output directory for this trajectory
        traj_output_dir = os.path.join(compare_dir, str(traj_num))
        os.makedirs(traj_output_dir, exist_ok=True)
        
        # Load mesh
        inp_file = os.path.join(dataset_dir, str(traj_num), 'Bump.inp')
        if not os.path.exists(inp_file):
            print(f"Warning: {inp_file} not found, skipping trajectory {traj_num}")
            continue
        
        nodes, elements = parse_inp_file(inp_file)
        mesh, num_nodes = create_pyvista_mesh(nodes, elements)
        print(f"Loaded mesh with {num_nodes} nodes and {len(elements)} elements")
        
        # Load HDF5 data
        h5_file = os.path.join(result_dir, f'{traj_num}.h5')
        if not os.path.exists(h5_file):
            print(f"Warning: {h5_file} not found, skipping trajectory {traj_num}")
            continue
        
        with h5py.File(h5_file, 'r') as f:
            predicted = f['predicteds'][:]  # Shape: (T, N, 4)
            target = f['targets'][:]        # Shape: (T, N, 4)
        
        num_timesteps = predicted.shape[0]
        print(f"Loaded data with {num_timesteps} timesteps")
        
        # Check if data dimensions match mesh
        if predicted.shape[1] != num_nodes:
            print(f"Warning: Number of nodes in data ({predicted.shape[1]}) doesn't match mesh ({num_nodes})")
            print("Attempting to continue anyway...")
        
        # Select timesteps to plot
        timesteps_to_plot = list(range(0, num_timesteps, timestep_interval))
        
        # Always include the last timestep
        if (num_timesteps - 1) not in timesteps_to_plot:
            timesteps_to_plot.append(num_timesteps - 1)
        
        print(f"Plotting timesteps: {timesteps_to_plot}")
        
        # Create plots for each selected timestep
        for t in timesteps_to_plot:
            print(f"  Creating plots for timestep {t}...")
            
            # Get data for this timestep
            pred_t = predicted[t]  # Shape: (N, 4)
            targ_t = target[t]     # Shape: (N, 4)
            
            # Create comparison plot
            plot_comparison(mesh, pred_t, targ_t, var_names, t, traj_output_dir, traj_num)
            
            # Create error map
            plot_error_maps(mesh, pred_t, targ_t, var_names, t, traj_output_dir, traj_num)
        
        # Create a summary statistics file
        stats_file = os.path.join(traj_output_dir, 'statistics.txt')
        with open(stats_file, 'w') as f:
            f.write(f"Trajectory {traj_num} Statistics\n")
            f.write("=" * 40 + "\n\n")
            
            for var_idx, var_name in enumerate(var_names):
                f.write(f"{var_name}:\n")
                
                # Calculate errors across all timesteps
                errors = np.abs(predicted[:, :, var_idx] - target[:, :, var_idx])
                
                f.write(f"  Mean Absolute Error: {np.mean(errors):.6e}\n")
                f.write(f"  Max Absolute Error: {np.max(errors):.6e}\n")
                f.write(f"  RMSE: {np.sqrt(np.mean(errors**2)):.6e}\n")
                
                # Relative error (avoid division by zero)
                target_var = target[:, :, var_idx]
                mask = np.abs(target_var) > 1e-10
                rel_errors = np.zeros_like(errors)
                rel_errors[mask] = errors[mask] / np.abs(target_var[mask])
                
                f.write(f"  Mean Relative Error: {np.mean(rel_errors[mask]):.6e}\n\n")
        
        print(f"Saved statistics to {stats_file}")
    
    print("\nAll trajectories processed successfully!")
    print(f"Results saved in '{compare_dir}' directory")

if __name__ == "__main__":
    main()
