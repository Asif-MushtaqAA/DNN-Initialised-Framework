import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from scipy.spatial import cKDTree
import scipy.io as sio
import matlab.engine
import subprocess
from scipy.interpolate import griddata
import shutil
import time

# Define the path to the folder containing modules
FliteonPy_dir = os.path.abspath(os.path.join('..', 'FLITEonPY'))
sys.path.append(FliteonPy_dir)

sdf_dir = os.path.abspath(os.path.join('..', 'DNN/UI'))
sys.path.append(sdf_dir)

dnn_dir = os.path.abspath(os.path.join('..', 'DNN/UI'))
sys.path.append(dnn_dir)

from sdf_generator import generate_sdf

from DNN_UI import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, model_init, main_inference # Need to call first three in main(console) everytime

from mesh_gen import mesh_gen
from save_mesh import save_mesh
from import_FLITE_data import import_FLITE_data

#1.SDF Generator

# SDF generator module imported

#.................................................................................................................................................
#1.5. DNN Inference

# DNN module imported

#.................................................................................................................................................
#2.Process Inferred Field
def load_inferred_field(airfoil_number, mach, aoa, inferred_data_folder):
    inferred_file_path = os.path.join(inferred_data_folder, f'{airfoil_number}_{mach}_{aoa}.npy')
    inferred_field = np.load(inferred_file_path)
    return inferred_field

def introduce_airfoil_geometry(inferred_field, sdf):
    # Set values to NaN for points inside the airfoil
    inferred_field[sdf < 0] = np.nan
    return inferred_field

def load_global_min_max():
    global_min_max_folder = './functions'
    global_min_path = os.path.join(global_min_max_folder, 'global_min.npy')
    global_max_path = os.path.join(global_min_max_folder, 'global_max.npy')
    
    global_min = np.load(global_min_path)
    global_max = np.load(global_max_path)
    
    return global_min, global_max

def denormalize_inferred_field(inferred_field, global_min, global_max):
    denormalized_field = np.empty_like(inferred_field)
    
    for i in range(inferred_field.shape[2]):  # Iterate over channels
        denormalized_field[:, :, i] = inferred_field[:, :, i] * (global_max[i] - global_min[i]) + global_min[i]
    
    # Ensure to ignore NaN values
    denormalized_field[np.isnan(inferred_field)] = np.nan
    return denormalized_field

# Plot the denormalized field for each channel
def plot_denormalized_field(denormalized_field, airfoil_number):
    channels = denormalized_field.shape[2]
    fig, axes = plt.subplots(1, channels, figsize=(20, 5))
    for i in range(channels):
        im = axes[i].contourf(denormalized_field[:, :, i], levels=200, cmap='turbo')
        fig.colorbar(im, ax=axes[i], label=f'Channel {i+1}')
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
        axes[i].grid(False)
    plt.suptitle(f'Denormalized Field for Airfoil {airfoil_number}')
    plt.tight_layout()
    plt.show()

def process_inferred_field(sdf_image, airfoil_number, mach, aoa, output_dir):
    
    start_m = time.time()
    
    inferred_field = load_inferred_field(airfoil_number, mach, aoa, output_dir)
    inferred_field = introduce_airfoil_geometry(inferred_field, sdf_image)
    global_min, global_max = load_global_min_max()
    denormalized_field = denormalize_inferred_field(inferred_field, global_min, global_max)
    
    end_m = time.time()
    elapsed_total = end_m - start_m
    print(f'Time taken for denormalization: {elapsed_total:.2f} seconds')
    print("Denormalized inferred field processed and ready for the next step.")
    #plot_denormalized_field(denormalized_field, airfoil_number) #...
    return denormalized_field
#.................................................................................................................................................
#3.Mesh generation
def load_data():
    flow_field_path = './functions/flow_field.txt'
    bound_data_path = './functions/bound_data.txt'
    psource_path = './functions/psource.txt'
    
    flow_field = np.loadtxt(flow_field_path)
    bound_data = np.loadtxt(bound_data_path).astype(int)
    psource = np.loadtxt(psource_path)
    
    return flow_field, bound_data, psource

def run_prepro():
    # Change directory to where CFD-related files are located
    os.chdir('./functions')
    
    prepro_exe = r'PrePro.exe'
    input_file = r'runPrePro.inp'

    if not os.path.exists(prepro_exe):
        raise FileNotFoundError(f"PrePro.exe not found at {prepro_exe}")

    # Open PrePro.exe process
    with subprocess.Popen([prepro_exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        # Read inputs from input file and feed them sequentially
        with open(input_file, 'r') as f:
            for line in f:
                input_value = line.strip()  
                process.stdin.write(input_value + '\n')
                process.stdin.flush() 

        process.stdin.close()

        stdout, stderr = process.communicate()

        # Check if process exited successfully
        if process.returncode != 0:
            print(f"PrePro.exe terminated with error: {stderr}")
        else:
            print("PrePro.exe execution completed successfully.")
            
    # Change directory back to original
    os.chdir('..')

def mesh_generation_workflow(airfoil_coords):

    # Load required data
    flow_field, bound_data, psource = load_data()
    
    # Concatenate coordinates and flow field data
    xy = np.vstack((airfoil_coords, flow_field))

    # Generate the mesh
    start_m= time.time()

    mesh = mesh_gen(xy, bound_data, alpha=0.8, psource=psource)

    end_m = time.time() 
    elapsed = end_m -start_m
    print(f'Time taken for mesh generation: {elapsed:.2f} seconds')

    if mesh is not None:
        save_mesh(mesh, './functions/mesh.dat', bound_data)
    else:
        print("Mesh generation failed; skipping save.")

    # Run prepro.exe to generate mesh.sol
    run_prepro()

    return mesh
#.................................................................................................................................................
#4.Generating Initialiser file
# Function to load coordinates from mesh.dat
def parse_mesh_coordinates(mesh_file):
    coordinates = []
    with open(mesh_file, 'r') as file:
        reading_coordinates = False
        for line in file:
            if line.strip() == 'coordinates':
                reading_coordinates = True
                continue
            if line.strip() == 'boundaries':
                break
            if reading_coordinates:
                parts = line.split()
                if len(parts) == 3:
                    element_id = int(parts[0])
                    x_coord = float(parts[1])
                    y_coord = float(parts[2])
                    coordinates.append([element_id, x_coord, y_coord])
    return np.array(coordinates)
    
# Function to extract grid values
def extract_grid_values(grid_values):
    return grid_values.reshape(-1, grid_values.shape[2])

# Function to map grid values to coordinates with element IDs
def map_grid_to_coordinates(grid_x, grid_y, grid_values, original_coords, x_range, y_range):
    flattened_grid_points = np.vstack([grid_x.ravel(), grid_y.ravel()]).T
    
    # Initialize an array to hold the mapped values and element IDs
    num_variables = grid_values.shape[1]
    mapped_values = np.full((len(original_coords), num_variables + 1), np.nan)  # +1 for element ID
    mapped_values[:, 0] = original_coords[:, 0]  # Set element IDs

    # Filter original coordinates to be within the specified range
    in_range = (original_coords[:, 1] >= x_range[0]) & (original_coords[:, 1] <= x_range[1]) & \
               (original_coords[:, 2] >= y_range[0]) & (original_coords[:, 2] <= y_range[1])
    valid_coords = original_coords[in_range]

    for i in range(len(valid_coords)):
        original_x, original_y = valid_coords[i, 1:3]
        distances = np.sqrt((flattened_grid_points[:, 0] - original_x)**2 + 
                             (flattened_grid_points[:, 1] - original_y)**2)
        nearest_idx = np.argmin(distances)
        # Find the index in the full mapped_values array
        original_index = np.where(original_coords[:, 0] == valid_coords[i, 0])[0][0]
        mapped_values[original_index, 1:] = grid_values[nearest_idx]
    
    return mapped_values, in_range

# Function to interpolate missing values
def interpolate_missing_values(coords, values):
    # Filter out NaN values
    valid_indices = ~np.isnan(values[:, 1:]).any(axis=1)
    valid_coords = coords[valid_indices, 1:3]
    valid_values = values[valid_indices, 1:]
    
    interpolated_values = np.copy(values)
    for i in range(values.shape[1] - 1):  # Skip the first column (IDs)
        interpolated_values[:, i + 1] = griddata(valid_coords, valid_values[:, i], coords[:, 1:3], method='nearest', fill_value=1)
    
    return interpolated_values

# Function to save mapped values
def save_mapped_values(original_coords, mapped_values, file_path):
    with open(file_path, 'w') as file:
        for i, coords in enumerate(original_coords):
            element_id = int(coords[0])
            id_format = f"{element_id:>5}"  # Right-align with 5-character wide field
            
            # Format values in scientific notation with 4 decimal places
            formatted_values = [f"{val: .4E}" for val in mapped_values[i, 1:]]
            
            # Join the formatted values with 3 spaces
            values_line = "   ".join(formatted_values)
            
            # Combine ID and values with 3 spaces in between
            line = f"{id_format}   {values_line}"
            
            file.write(line + '\n')
            
# Function to plot the mapped values for all channels
def plot_mapped_values(coordinates, mapped_values, title='Mapped Values'):
    num_channels = mapped_values.shape[1] - 1
    fig, axes = plt.subplots(1, num_channels, figsize=(20, 5))
    if num_channels == 1:
        axes = [axes]
    for i in range(num_channels):
        sc = axes[i].scatter(coordinates[:, 1], coordinates[:, 2], c=mapped_values[:, i + 1], cmap='turbo')
        plt.colorbar(sc, ax=axes[i])
        axes[i].set_title(f'Channel {i+1}')
        axes[i].set_xlabel('X')
        axes[i].set_ylabel('Y')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Main function to process the field mapping
def process_field_mapping(grid_field, grid_x, grid_y, x_range, y_range):
    
    mesh_file = './functions/mesh.dat'
    output_file = './functions/dnninit.res'
    
    start_m = time.time()
    # Extract the coordinates
    coordinates = parse_mesh_coordinates(mesh_file)
    
    # Extract grid values
    grid_values = extract_grid_values(grid_field)
    
    # Map grid values to original coordinates
    mapped_values, in_range = map_grid_to_coordinates(grid_x, grid_y, grid_values, coordinates, x_range, y_range)
    
    # Plot the mapped values for all channels
    #plot_mapped_values(coordinates[in_range], mapped_values[in_range], title='Mapped Values ROI Region') #...
    
    # Interpolate missing values
    interpolated_values = interpolate_missing_values(coordinates, mapped_values)
    
    # Plot the interpolated values for all channels
    #plot_mapped_values(coordinates, interpolated_values, title='Mapped Values Full Region') #...
    
    # Save the results
    save_mapped_values(coordinates, interpolated_values, output_file)
    
    end_m = time.time()
    elapsed_total = end_m - start_m
    print(f'Time taken for creating initialiser file: {elapsed_total:.2f} seconds')
    print("Field mapping completed and saved.")
#.................................................................................................................................................    
#5.FLITE2D Solver
def run_solver_and_generate_plots(mach, aoa, mesh):
    # Change directory to where CFD-related files are located
    os.chdir('./functions')
    
    start_s = time.time()
    
    # Solver.exe - Run for each combination of mach and aoa
    solver_exe = r'Solver.exe'
    inp_file = r'runSolver.inp'
    solverinp_file = r'solver.inp'
    
    if not os.path.exists(solver_exe):
        raise FileNotFoundError(f"Solver.exe not found at {solver_exe}")

    # Modify input file 'solver.inp' based on mach and aoa
    with open(solverinp_file, 'r') as f:
        solver_input = f.readlines()
    
    # Update mach number and AoA in the input file
    for line_idx, line in enumerate(solver_input):
        if 'ivd%alpha' in line:
            solver_input[line_idx] = f" ivd%alpha = {aoa:.2f},\n"
        elif 'ivd%MachNumber' in line:
            solver_input[line_idx] = f" ivd%MachNumber = {mach:.2f},\n"  # Use 2 decimal place for mach

    with open(solverinp_file, 'w') as f:
        f.writelines(solver_input)
    
    # Run Solver.exe
    with subprocess.Popen([solver_exe], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        # Read inputs from input file and feed them sequentially
        with open(inp_file, 'r') as f:
            for line in f:
                input_value = line.strip() 
                process.stdin.write(input_value + '\n')
                process.stdin.flush() 

        process.stdin.close()

        stdout, stderr = process.communicate()

        # Check if process exited successfully
        if process.returncode != 0:
            print(f"Solver.exe terminated with error: {stderr}")
        else:
            print("Solver.exe execution completed successfully.")
    
    end_s = time.time()
    elapsed_total = end_s - start_s
    print(f'Time taken for solver run: {elapsed_total:.2f} seconds')
    
    # Change directory back to original
    os.chdir('..')
    
    # Import results using import_FLITE_data function
    results, residual = import_FLITE_data('./functions/solverout.rsd', './functions/solverout.res', mesh)

def read_solver_output_and_compute_cl_cd(alpha, solverout_path = './functions/solverout.rsd'):
    # Reading the solverout.rsd file
    with open(solverout_path, 'r') as file:
        lines = file.readlines()
    
    # Checking that there are at least two lines
    if len(lines) < 2:
        raise ValueError("The solverout.rsd file does not contain enough data lines.")
    
    # Get the second to last line (last line is empty)
    second_last_line = lines[-1].strip()
    if not second_last_line:
        raise ValueError("The second last line is unexpectedly empty.")
    
    # Split the line into components and extract CY and CX
    parts = second_last_line.split()
    CY = float(parts[2])
    CX = float(parts[3])
    print(f"CY: {CY}, CX: {CX}")
    
    # Compute CL and CD
    # Convert angle of attack from degrees to radians
    alpha_rad = np.radians(alpha)

    # Calculate lift coefficient (CL) and drag coefficient (CD)
    cl = (-CY * np.cos(alpha_rad)) - (CX * np.sin(alpha_rad))
    cd = (-CY * np.sin(alpha_rad)) + (CX * np.cos(alpha_rad))
    print(f"CL: {cl}, CD: {cd}")
    
    return cl, cd

def move_results_files(airfoil_number, mach, aoa):
    
    source_folder = './functions'
    results_folder = './results'
    
    # Create a new folder for results
    result_folder_name = f"{airfoil_number}_{mach}_{aoa}"
    result_folder_path = os.path.join(results_folder, result_folder_name)
    
    if not os.path.exists(result_folder_path):
        os.makedirs(result_folder_path)
    
    # List of files to move
    files_to_move = [
        'data.mat',
        'mesh.dat',
        'mesh.sol',
        'solverout.res',
        'solverout.rsd',
        'dnninit.res'
    ]
    
    for file_name in files_to_move:
        src = os.path.join(source_folder, file_name)
        dest = os.path.join(result_folder_path, file_name)
        if os.path.exists(src):
            shutil.move(src, dest)
            
    print(f"Files have been moved to folder {result_folder_path} successfully.")    
    
def workflow(airfoil_number, mach, aoa, sdf_path = './sdf', geo_path = './data_geometry', output_dir = './inferred_data'): 
    # Timing measurement
    start_total = time.time()
    
    airfoil_coords, sdf_image, grid_x, grid_y, x_range, y_range = generate_sdf(airfoil_number, sdf_path, geo_path)
    
    model = model_init()
    
    main_inference(airfoil_number, mach, aoa, model, sdf_path, output_dir)
    
    denormalized_field = process_inferred_field(sdf_image, airfoil_number, mach, aoa, output_dir)
    
    mesh = mesh_generation_workflow(airfoil_coords)
    
    process_field_mapping(denormalized_field, grid_x, grid_y, x_range, y_range)
    
    run_solver_and_generate_plots(mach, aoa, mesh)
    
    cl, cd = read_solver_output_and_compute_cl_cd(aoa)
    
    move_results_files(airfoil_number, mach, aoa)
    
    print(f"Workflow for airfoil {airfoil_number} with Mach {mach} and AoA {aoa} completed.")
    
    # Total execution time
    end_total = time.time()
    elapsed_total = end_total - start_total
    print(f'Total time taken: {elapsed_total:.2f} seconds')
    
    return cl, cd
#Example Implementation in console
#from DNN_FLITEonPY import ResidualBlock, ChannelSpecificDecoder, EncoderDecoderCNN, workflow
#workflow(58,0.6,2)