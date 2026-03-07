# Contributing to InverseBench

Thank you for your interest in contributing to InverseBench! This document provides guidelines and instructions for contributing to this project.

## Table of Contents

- [How Can I Contribute?](#how-can-i-contribute)
- [Architecture Overview](#architecture-overview)
- [Adding a New Inverse Problem](#adding-a-new-inverse-problem)
- [Adding a New Algorithm](#adding-a-new-algorithm)
- [Adding a New Evaluator](#adding-a-new-evaluator)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)

## How Can I Contribute?

There are several ways to contribute to InverseBench:

1. **Add a new inverse problem** -- Extend the benchmark with a new scientific inverse problem by implementing a forward operator, dataset, evaluator, and Hydra config.
2. **Add a new algorithm** -- Implement a new plug-and-play diffusion prior algorithm or domain-specific baseline.
3. **Improve existing implementations** -- Fix bugs, improve performance, or enhance numerical stability of existing operators or algorithms.
4. **Improve documentation** -- Clarify existing docs, add examples, or write tutorials.
5. **Report issues** -- File bug reports or suggest enhancements via GitHub Issues.

## Architecture Overview

InverseBench has a modular design built around four core abstractions. Understanding them is essential before contributing.

### Core Abstractions

| Component | Base Class | Location | Responsibility |
|-----------|-----------|----------|---------------|
| **Forward Operator** | `BaseOperator` | `inverse_problems/base.py` | Defines the physics-based forward model `y = A(x) + noise` |
| **Algorithm** | `Algo` | `algo/base.py` | Solves the inverse problem given an observation and a diffusion prior |
| **Evaluator** | `Evaluator` | `eval.py` | Computes problem-specific reconstruction quality metrics |
| **Diffusion Model** | DDPM / UNet | `models/` | Provides the learned prior used by algorithms |

### Data Flow

```
Configuration (Hydra YAML)
        |
        v
  +------------------+      +------------------+      +------------------+
  | Forward Operator  | <--- |    Algorithm      | ---> |    Evaluator      |
  | (inverse_problems)|      |     (algo/)       |      |    (eval.py)      |
  +------------------+      +------------------+      +------------------+
        |                          |
        | y = A(x) + noise         | x_recon = algo.inference(y)
        v                          |
  +------------------+      +------------------+
  |   Observation     |      | Diffusion Model   |
  |       y           |      |    (models/)       |
  +------------------+      +------------------+
```

### Directory Layout

```
InverseBench/
  algo/               # Algorithm implementations (14 algorithms)
    base.py           #   Abstract base class: Algo
    dps.py            #   Diffusion Posterior Sampling
    daps.py           #   Diffusion Annealing Posterior Sampling
    ...               #   (see algo/ for the full list)
  inverse_problems/   # Forward operators for each scientific problem
    base.py           #   Abstract base class: BaseOperator
    acoustic.py       #   Full Waveform Inversion (seismology)
    navier_stokes.py  #   2D Navier-Stokes (fluid dynamics)
    inverse_scatter.py#   Linear inverse scattering (optical tomography)
    multi_coil_mri.py #   Multi-coil MRI (medical imaging)
    blackhole.py      #   Black hole imaging (EHT)
  models/             # Diffusion model architectures
    ddpm.py           #   DDPM + UNet implementation
    unets.py          #   UNet variants
    precond.py        #   EDM-style preconditioning
    e2e/              #   End-to-end models for MRI baselines
  training/           # Training utilities
    dataset.py        #   Dataset classes (LMDB, ImageFolder, MRI)
    loss.py           #   Loss functions (VP, VE, EDM, PSNR, SSIM)
  utils/              # Shared utilities
    scheduler.py      #   Diffusion noise schedules and timesteps
    helper.py         #   EMA, logging, model loading
  configs/            # Hydra configuration files
    config.yaml       #   Default top-level config
    algorithm/        #   Per-algorithm configs
    problem/          #   Per-problem configs
    pretrain/         #   Pre-trained model configs
    sweep/            #   Wandb hyperparameter sweep configs
  eval.py             # Evaluator implementations
  main.py             # Inference + evaluation entry point
  train.py            # Diffusion model training entry point
```

## Adding a New Inverse Problem

To add a new inverse problem, you need to implement four components:

### 1. Forward Operator

Create a new file in `inverse_problems/` that subclasses `BaseOperator`:

```python
# inverse_problems/my_problem.py
from inverse_problems.base import BaseOperator

class MyOperator(BaseOperator):
    def __init__(self, sigma_noise=0.0, unnorm_shift=0.0, unnorm_scale=1.0, device='cuda'):
        super().__init__(sigma_noise, unnorm_shift, unnorm_scale, device)
        # Initialize problem-specific parameters (e.g., physics simulator)

    def forward(self, inputs, **kwargs):
        """
        Apply the forward model A(x).

        Args:
            inputs: torch.Tensor of shape (batch_size, C, H, W), normalized.

        Returns:
            Measurements: torch.Tensor of shape (batch_size, ...).
        """
        # Implement the forward physics model
        ...

    def loss(self, pred, observation, **kwargs):
        """
        Optional: override if L2 loss is not appropriate for your problem.
        """
        ...
```

Key considerations:
- `forward()` receives **normalized** inputs (in the diffusion model's input range).
- Use `unnormalize()` inside `forward()` if the physics simulation needs physical units.
- Override `gradient()` if `torch.autograd.grad` does not work with your simulator (e.g., external solvers).
- Implement `close()` if your operator holds external resources (e.g., Dask cluster for FWI).

### 2. Dataset

Add your dataset class to `training/dataset.py` or create a new file. The dataset must return dictionaries with a `'target'` key:

```python
def __getitem__(self, idx):
    return {'target': tensor_of_shape_CHW}
```

### 3. Evaluator

Add an evaluator to `eval.py` that subclasses `Evaluator`:

```python
class MyProblemEvaluator(Evaluator):
    def __init__(self, forward_op=None):
        metric_list = {
            'my_metric': my_metric_function,  # (pred, target) -> scalar or tensor
        }
        super().__init__(metric_list, forward_op=forward_op)

    def __call__(self, pred, target, observation=None):
        metric_dict = {}
        for name, func in self.metric_list.items():
            val = func(pred, target).mean().item()
            metric_dict[name] = val
            self.metric_state[name].append(val)
        return metric_dict
```

### 4. Hydra Configuration

Create a new config file in `configs/problem/`:

```yaml
# configs/problem/my_problem.yaml
name: my_problem
exp_dir: results/my_problem
prior: /path/to/pretrained/model.pt

model:
  _target_: inverse_problems.my_problem.MyOperator
  sigma_noise: 0.05

data:
  _target_: training.dataset.MyDataset
  data_path: /path/to/data
  id_list: "1-10"

evaluator:
  _target_: eval.MyProblemEvaluator
```

Then run inference with:

```bash
python3 main.py problem=my_problem algorithm=dps pretrain=my_model
```

## Adding a New Algorithm

Create a new file in `algo/` that subclasses `Algo`:

```python
# algo/my_algo.py
from algo.base import Algo

class MyAlgorithm(Algo):
    def __init__(self, net, forward_op, my_param=1.0):
        super().__init__(net, forward_op)
        self.my_param = my_param

    def inference(self, observation, num_samples=1, **kwargs):
        """
        Solve the inverse problem.

        Args:
            observation: Measurement tensor from the forward operator.
            num_samples: Number of reconstruction samples to generate.

        Returns:
            Reconstructions: torch.Tensor of shape (num_samples, C, H, W).
        """
        # Implement the reconstruction algorithm
        ...
```

Then add a Hydra config in `configs/algorithm/`:

```yaml
# configs/algorithm/my_algo.yaml
name: my_algo
method:
  _target_: algo.my_algo.MyAlgorithm
  my_param: 1.0
```

Tips:
- Use `self.net` to access the pre-trained diffusion model for denoising.
- Use `self.forward_op` to compute forward model evaluations and gradients.
- Use `utils/scheduler.py` for diffusion noise schedules and timestep discretization.
- Return results in the **normalized** space (the pipeline calls `unnormalize()` after inference).

## Adding a New Evaluator

If your new problem requires custom metrics, subclass `Evaluator` in `eval.py`. The key interface methods are:

- `__call__(pred, target, observation)` -- Compute per-sample metrics and append to `self.metric_state`.
- `compute()` -- Aggregate all tracked metrics into mean and std (inherited from base class).

## Development Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/devzhk/InverseBench.git
   cd InverseBench
   ```

2. **Install dependencies:**

   ```bash
   # Using uv (recommended)
   uv sync
   source .venv/bin/activate

   # Or using conda
   conda env create -f env.yaml
   conda activate inversebench
   ```

3. **Verify installation:**

   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

4. **Run tests** (if applicable):

   ```bash
   pytest
   ```

## Code Style

- Follow the existing code conventions in the repository.
- Use type hints for function signatures where practical.
- Keep forward operators, algorithms, and evaluators in their respective directories.
- Use Hydra `_target_` for all instantiable classes so they integrate with the config system.
- Prefer `torch` operations over NumPy for GPU compatibility.
- Document non-obvious physics or mathematical details with inline comments or docstrings.

## Submitting Changes

1. **Fork** the repository and create a feature branch:

   ```bash
   git checkout -b feature/my-new-problem
   ```

2. **Make your changes** following the guidelines above.

3. **Test your changes** by running inference on a small subset of data:

   ```bash
   python3 main.py problem=my_problem algorithm=dps pretrain=my_model problem.data.id_list="1-2"
   ```

4. **Commit** with a clear message:

   ```bash
   git commit -m "Add [problem/algorithm name]: brief description"
   ```

5. **Open a Pull Request** against the `main` branch with:
   - A description of what you added or changed.
   - Relevant benchmark results (if adding a new problem or algorithm).
   - Any new dependencies and why they are needed.

## Reporting Issues

When reporting a bug, please include:

- Python version and OS.
- GPU model and CUDA version.
- Steps to reproduce the issue.
- Full error traceback.
- The Hydra config used (or the command line invocation).

For feature requests, describe the scientific problem or algorithm you'd like to see supported and link to relevant papers if applicable.
