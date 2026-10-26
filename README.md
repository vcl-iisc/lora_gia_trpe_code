# Breaching - WACV Submission Codebase

This repository contains the codebase for evaluating privacy breaches in federated learning models, submitted as supplementary material for WACV. It provides implementations of privacy attacks (like GIAS), case configurations, and execution scripts.

This repository is derived from the original [Breaching](https://github.com/JonasGeiping/breaching) repository

## Installation

1. Create a virtual environment and activate it:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### Reproducing Attack Benchmark

To simulate and reproduce the multi-round attack scenarios detailed in the paper, you can use the provided bash script. This script executes a multi-round federated learning benchmark:
```bash
chmod +x run_attack.sh
bash run_attack.sh
```

The script configures various hyperparameters like `STEP_SIZE`, `OBJ_SCALE`, `TV_SCALE`, and `LORA_RANK`. Feel free to modify the environment variables inside or before calling the script to test different configurations.

For a full sweep across hyperparameters, you can refer to `hyperparamsearch.sh`.

## Configuration Structure

The configuration files are built using Hydra and are located under `breaching/config/`. The core benchmark logic can be found in `benchmark_breaches_multirounds.py` and the core attack implementations under `breaching/attacks`.
