import h5py
import numpy as np
import os
import sys


def downsample_h5_data(input_path, output_path, downsample_factor):
    """
    Downsamples the time steps in the HDF5 dataset by the given factor.

    Parameters:
    - input_path: 
    - output_path: 
    - downsample_factor: int, the factor by which to downsample the time steps (e.g., 10)
    """
    os.makedirs(output_path, exist_ok=True)
    total_timesteps = 600  # Original number of time steps in the data
    N = downsample_factor  # Downsampling factor

    # Check if downsample_factor divides total_timesteps evenly
    if total_timesteps % N != 0:
        print(
            f"Error: downsample_factor {N} does not evenly divide total_timesteps {total_timesteps}."
        )
        print("Please choose a downsample_factor that divides 600 evenly.")
        sys.exit(1)

    # Calculate the indices for uniform downsampling
    indices = np.arange(N - 1, total_timesteps, N)  # Start from the Nth fine timestep
    print("Is the last fine timestep included?", (total_timesteps - 1) in indices)

    for split in ["train", "valid", "test"]:
        input_file = os.path.join(input_path, split + ".h5")
        output_file = os.path.join(
            output_path, split + ".h5"
        )  # Save with the same name
        print(f"Processing {input_file} -> {output_file}")

        with h5py.File(input_file, "r") as f_in, h5py.File(output_file, "w") as f_out:
            # For each trajectory
            for traj_key in f_in.keys():
                print(f"Processing trajectory {traj_key}")
                group_in = f_in[traj_key]
                group_out = f_out.create_group(traj_key)

                # For each dataset in the trajectory
                for dataset_name in group_in.keys():
                    data = group_in[dataset_name][...]

                    # Both static and dynamic features have time dimension
                    # Downsample along the first axis (time)
                    data_downsampled = data[indices, ...]

                    group_out.create_dataset(dataset_name, data=data_downsampled)
        print(f"Finished processing {split}")


if __name__ == "__main__":
    input_path = "dataset/data"  # Path where original HDF5 files are located
    output_path = (
        "dataset/data_downsampled"  # Path where downsampled HDF5 files will be saved
    )
    downsample_factor = 5

    downsample_h5_data(input_path, output_path, downsample_factor)

    with h5py.File("dataset/data_downsampled/train.h5", "r") as f:
        traj = f["0"]  # Access the first trajectory
        print(traj["velocity"].shape)  # (60, num_nodes, 2) if downsample_factor is 10
        print(traj["pressure"].shape)  # (60, num_nodes, 1)
        print(traj["pos"].shape)  # (60, num_nodes, 2)
