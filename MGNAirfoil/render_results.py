import h5py
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.tri as tri
import cv2
from tqdm import tqdm
import glob
import os

# Adjust font sizes and styles
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 12,
        "font.serif": ["DejaVu Sans"],
    }
)


# Specify parameters
timestep_size = 0.04
n_steps = None  # Number of timesteps (determined from data if None)
skip = 1  # Number of steps to skip between frames
max_files = None  # Number of files to process (set to None to process all files)


result_files = glob.glob("result/*.h5")
result_files.sort(
    key=lambda x: int(os.path.basename(x).replace("result", "").replace(".h5", ""))
)
os.makedirs("videos", exist_ok=True)

# Limit the number of files to process
if max_files is not None:
    result_files = result_files[:max_files]

for index, file in enumerate(result_files):

    # Read data from .h5 file
    with h5py.File(file, "r") as f:
        predicted_velocity = f["predicted_velocity"][:]  # Shape: (n_steps, n_nodes, n_features)
        target_velocity = f["target_velocity"][:] 
        crds = f["coordinates"][:]  # Shape: (n_nodes, 2)
        face = f["face"][:]  # Shape: (3, n_faces) or (n_faces, 3)

    # Transpose face if necessary
    if face.shape[0] == 3:
        face = face.T  # Now face.shape is (n_faces, 3)

    # Create triangulation using face data
    triang = tri.Triangulation(crds[:, 0], crds[:, 1], triangles=face)

    # Determine n_steps from data if not specified
    if n_steps is None:
        n_steps = min(predicted_velocity.shape[0], target_velocity.shape[0])

    n_frames = (n_steps + skip - 1) // skip

    # Compute velocity magnitudes for global v_min and v_max
    target_magnitudes = np.linalg.norm(target_velocity, axis=-1).flatten()
    predicted_magnitudes = np.linalg.norm(predicted_velocity, axis=-1).flatten()
    v_max = np.max(target_magnitudes)
    v_min = np.min(target_magnitudes)

    # Compute global v_min and v_max for absolute difference
    abs_diff_magnitudes = np.abs(target_magnitudes - predicted_magnitudes)
    diff_vmax = np.max(abs_diff_magnitudes)
    diff_vmin = np.min(abs_diff_magnitudes)

    # Set up video writer
    # Create a dummy figure to get the size
    fig = plt.figure(figsize=(6, 16), dpi=200)
    plt.close(fig)
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()

    # Ensure frame dimensions are even numbers
    width = int((width // 2) * 2)
    height = int((height // 2) * 2)

    file_name = f"videos/output{index}.avi"
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(file_name, fourcc, 10.0, (width, height))

    def render(i):
        step = i * skip
        target = target_velocity[step]
        predicted = predicted_velocity[step]

        fig = plt.figure(figsize=(6, 16), dpi=200)

        # Create a GridSpec with appropriate heights
        from matplotlib.gridspec import GridSpec

        gs = GridSpec(
            nrows=10,
            ncols=1,
            height_ratios=[1.6, 1.6, 0.0, 0.1, 0.4, 1.6, 0.0, 0.1, 0.0, 0.0],
            hspace=0.0,
        )

        # Define rectangular zoom region around the airfoil
        x_min, x_max = -0.6, 1.8
        y_min, y_max = -0.8, 0.8

        # First subplot: Target
        ax1 = fig.add_subplot(gs[0, 0])
        target_v = np.linalg.norm(target, axis=-1)
        tcf1 = ax1.tripcolor(
            triang,
            target_v,
            shading="flat",
            cmap="viridis",
            vmin=v_min,
            vmax=v_max,
            edgecolors="0.8",
        )
        ax1.set_title(f"Target\nTime @ {(step + 2) * timestep_size:.2f} s", fontsize=16)
        ax1.set_aspect("equal")
        ax1.axis("off")
        ax1.set_adjustable("box")
        # Set zoomed-in region
        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)

        # Second subplot: Prediction
        ax2 = fig.add_subplot(gs[1, 0])
        predicted_v = np.linalg.norm(predicted, axis=-1)
        tcf2 = ax2.tripcolor(
            triang,
            predicted_v,
            shading="flat",
            cmap="viridis",
            vmin=v_min,
            vmax=v_max,
            edgecolors="0.8",
        )
        ax2.set_title(
            f"Prediction\nTime @ {(step + 2) * timestep_size:.2f} s", fontsize=16
        )
        ax2.set_aspect("equal")
        ax2.axis("off")
        ax2.set_adjustable("box")
        # Set zoomed-in region
        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)

        # Colorbar for target and prediction (shared)
        cax1 = fig.add_subplot(gs[3, 0])
        cbar1 = fig.colorbar(
            tcf1,
            cax=cax1,
            orientation="horizontal",
        )
        cbar1.ax.set_xlabel("Velocity Magnitude", fontsize=14)
        cbar1.ax.tick_params(labelsize=14)

        # Third subplot: Absolute Difference
        ax3 = fig.add_subplot(gs[5, 0])
        abs_diff_v = np.abs(target_v - predicted_v)
        tcf3 = ax3.tripcolor(
            triang,
            abs_diff_v,
            shading="flat",
            cmap="viridis",
            vmin=diff_vmin,
            vmax=diff_vmax,
            edgecolors="0.8",
        )
        ax3.set_title(
            f"Absolute Difference\nTime @ {(step + 2) * timestep_size:.2f} s",
            fontsize=16,
        )
        ax3.set_aspect("equal")
        ax3.axis("off")
        ax3.set_adjustable("box")
        # Set zoomed-in region
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)

        # Colorbar for absolute difference
        cax2 = fig.add_subplot(gs[7, 0])
        cbar2 = fig.colorbar(
            tcf3,
            cax=cax2,
            orientation="horizontal",
        )
        cbar2.ax.set_xlabel("Absolute Difference", fontsize=14)
        cbar2.ax.tick_params(labelsize=14)

        # Adjust the layout
        plt.subplots_adjust(left=0.05, right=0.95, top=0.98, bottom=0.05, hspace=0.0)

        # Convert figure to image using buffer_rgba()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        img = img[:, :, :3]  # Discard alpha channel
        plt.close(fig)

        # Resize image to match video dimensions
        img = cv2.resize(img, (width, height))

        # Write frame to video
        out.write(cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    # Render frames
    for i in tqdm(range(n_frames), total=n_frames):
        step = i * skip
        if step < n_steps:
            render(i)

    out.release()
    print(f"Video {file_name} saved")
