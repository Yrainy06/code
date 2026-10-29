# Theory-Guided Training for Spatio-Temporal Neural PDE Surrogates

Code accompanying **Theory-Guided Training for Spatio-Temporal Neural PDE Surrogates with Stable Long-Horizon Rollouts**.

The method combines a graph neural network for spatial processing with a recurrent neural network for temporal modeling. Training uses teacher forcing (TF), followed by autoregressive (AR) multistep training.

This README documents the **2D Euler implementation in `cGnnCodes/`**. Run the commands below from that directory. The repository also contains files at its root; the commands here use the files in `cGnnCodes/` consistently.

## 1. Download the code

```bash
git clone https://github.com/Yrainy06/code.git
cd code/cGnnCodes
```

## 2. Environment

The current training and rollout scripts require an NVIDIA GPU with CUDA support: they contain explicit `.cuda()` calls, and the rollout scripts call `torch.cuda.set_device`. AutoDL's no-GPU mode can be used to transfer or edit files, but cannot run these scripts unchanged.

Install the following packages in a dedicated Python environment:

| Package | Purpose |
| --- | --- |
| `torch` | Neural networks and GPU computation |
| `torch_geometric` | Graph data, transformations, and loading |
| `torch_scatter` | Scatter operations imported by the GNN blocks |
| `numpy`, `h5py`, `tqdm` | Arrays, HDF5 files, and progress reporting |
| `matplotlib`, `pyvista` | Optional visualization |

First install a CUDA-enabled PyTorch build using the [official PyTorch installation instructions](https://pytorch.org/get-started/locally/). Then install the Python dependencies:

```bash
python -m pip install torch_geometric numpy h5py tqdm
```

Install `torch_scatter` using the wheel matching your PyTorch and CUDA versions, following the [official PyG installation instructions](https://pytorch-geometric.readthedocs.io/en/stable/install/installation.html). It is required by this code even though it is optional for some other PyG applications.

For visualization:

```bash
python -m pip install matplotlib pyvista
```

Check the environment:

```bash
python -c "import torch, torch_geometric, torch_scatter, h5py; print('PyTorch:', torch.__version__); print('PyG:', torch_geometric.__version__); print('CUDA build:', torch.version.cuda); print('GPU available:', torch.cuda.is_available())"
```

An exact tested environment and dependency lock file have not yet been supplied in this repository. To record an existing working environment, run `python -m pip freeze > requirements-tested.txt`.

## 3. Dataset preparation

The two dataset files, `train.h5` and `test.h5`, are distributed separately and are **not included in this Git repository**.

**Dataset download link: TODO — insert the public dataset download URL before distributing these instructions as a complete reproduction guide.**

Place the files at these paths relative to the repository root:

| File | Path |
| --- | --- |
| Training data | `cGnnCodes/dataset/data_downsampled/train.h5` |
| Test data | `cGnnCodes/dataset/data_downsampled/test.h5` |

Create the directory if necessary:

```bash
mkdir -p dataset/data_downsampled
```

Each HDF5 file contains trajectory groups. The loader in `dataset/fpcMulti.py` expects the keys `pos`, `edges`, `node_type`, `pres`, `rho`, `v1`, `v2`, and `Mach` within each trajectory. The existing loader indexes all of these fields by time, so replacement datasets must preserve the same layout, including the time axis of mesh and node metadata.

The raw node feature order is `[node_type, rho, v1, v2, pres, Mach]`. The predicted physical variables are `[rho, v1, v2, pres]`.

## 4. Main files

All paths in this table are relative to `cGnnCodes/`.

| File or directory | Purpose |
| --- | --- |
| `dataset/fpcMulti.py` | HDF5 loading, multistep training samples, and rollout samples |
| `model/model.py`, `model/blocks.py` | Spatial graph network |
| `model/simulator.py` | GNN simulator |
| `model/simulator_LSTM.py` | GNN with LSTM or GRU temporal processing |
| `trainMultiScheduled_LSTM.py` | Two-stage TF → AR training for the recurrent model |
| `trainMultiScheduled.py` | Two-stage training for the GNN model |
| `trainMulti_LSTM.py`, `trainMulti.py` | Additional training entry points |
| `scheduled_1.py`, `scheduled_2.py`, `scheduled_3.py` | Scheduled-sampling experiments |
| `rollout_LSTM.py`, `rollout.py` | Autoregressive evaluation |
| `extract_frames.py` | Field visualization using rollout results and original mesh files |
| `utils/` | Normalization, noise, loss, node types, and graph utilities |
| `checkpoint/simulator.pth` | Included checkpoint; use it only with its matching model configuration |

## 5. Two-stage recurrent-model training

**Before running training, review the final lines of the training script.** The uploaded `trainMultiScheduled_LSTM.py`, `trainMultiScheduled.py`, `trainMulti_LSTM.py`, and `trainMulti.py` call `/usr/bin/shutdown` after training. Comment out `os.system("/usr/bin/shutdown")` if you do not want the machine to shut down automatically.

From `cGnnCodes/`, run:

```bash
python trainMultiScheduled_LSTM.py --save_root checkpoint_LSTM_6_6
```

This uses the LSTM configuration currently selected in that script. Its main settings are defined in the source:

| Setting | Current value in `trainMultiScheduled_LSTM.py` |
| --- | --- |
| TF budget, `epochs_stage1` | 6 |
| AR budget, `epochs_stage2` | 6 |
| Batch size | 2 |
| Trajectory length, `traLength` | 80 |
| Training window, `numSteps` | 5 |
| Message-passing layers | 12 |
| TF learning rate | `1e-4` |
| AR learning rate | `1e-5` |
| TF noise, `[rho, v1, v2, pres]` | `[2e-2, 2e-2, 1e-2, 2e-2]` |
| AR noise, `[rho, v1, v2, pres]` | `[2e-3, 2e-3, 1e-3, 2e-3]` |

TF adds noise at each step of a training window. AR adds noise at the start of the window and feeds predicted states back at subsequent steps. Constrained nodes receive the boundary values supplied by the dataset.

Checkpoints are saved periodically to `checkpoint_LSTM_6_6/simulator.pth`. The current script does not make a separate unconditional final save after the last batch.

### LSTM versus GRU

`model/simulator_LSTM.py` supports both cells through `use_gru`:

```python
# LSTM
simulator = Simulator(
    message_passing_num=12,
    node_input_size=6,
    edge_input_size=5,
    device=device,
    use_gru=False,
)
```

To train GRU, set `use_gru=True` in the model construction in `trainMultiScheduled_LSTM.py` and use a separate output directory, for example:

```bash
python trainMultiScheduled_LSTM.py --save_root checkpoint_GRU_6_6
```

The filename alone does not identify the recurrent cell. For example, the uploaded `trainMulti_LSTM.py` explicitly selects `use_gru=True`, while `trainMultiScheduled_LSTM.py` uses the default LSTM. Match the cell and all architecture settings when loading a checkpoint.

## 6. Autoregressive rollout

After training the default LSTM configuration above, evaluate it using:

```bash
python rollout_LSTM.py \
  --gpu 0 \
  --model_dir checkpoint_LSTM_6_6/simulator.pth \
  --test_split test \
  --rollout_num 20 \
  --total_steps 79
```

Use a `--rollout_num` no larger than the number of trajectories in `test.h5`. For GRU evaluation, also set `use_gru=True` in the model construction in `rollout_LSTM.py` and pass the corresponding GRU checkpoint.

The current `rollout_LSTM.py` resets the recurrent state every five prediction steps, disables training noise through evaluation mode, and assigns dataset boundary values before feedback. Its `--total_steps` argument sets the progress-bar total; it does **not** truncate the iteration. The actual number of predictions is the number of stored trajectory states minus one. An 80-state trajectory therefore gives 79 predictions.

Outputs are written relative to `cGnnCodes/`:

| Output | Contents |
| --- | --- |
| `result/1.h5`, `result/2.h5`, … | `predicteds` and `targets`, with shape `[prediction_steps, nodes, 4]` |
| `metrics/per_trajectory_metrics.h5` | Per-trajectory cumulative and overall RMSE arrays |
| `metrics/average_metrics.h5` | Metrics averaged across trajectories |

Variable order is `[rho, v1, v2, pres]`. The current evaluation averages squared errors over all nodes, including constrained nodes. The output named `avg_one_step_rmse` is computed over the autoregressive trajectory; its name does not mean an independently teacher-forced evaluation. Aggregate metrics are arithmetic averages of the per-trajectory RMSE values.

## 7. Visualization

`extract_frames.py` reads `result/<trajectory_number>.h5` together with the original mesh at `Dataset/<trajectory_number>/Bump.inp`. The two HDF5 datasets alone do not provide these external mesh files.

After preparing those mesh files, review the path settings in the script and run:

```bash
python extract_frames.py
```

## 8. Reproduction notes

- Run scripts from `cGnnCodes/` so that relative paths and local imports resolve correctly.
- Select the intended TF/AR budgets, recurrent cell, architecture, and output directory explicitly; script defaults differ between experiments.
- The supplied checkpoint's training configuration should be documented before it is used to claim reproduction of a particular paper result.
- This guide was checked against the uploaded source. End-to-end training and evaluation require the separately distributed datasets and a compatible GPU environment.

## Associated manuscript

Xiaoyu Peng, Jingrun Chen, and Jiamin Jiang. *Theory-Guided Training for Spatio-Temporal Neural PDE Surrogates with Stable Long-Horizon Rollouts*.

Publication details and a formal citation can be added when available.
