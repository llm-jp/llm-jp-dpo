# LLM-jp DPO (Direct Preference Optimization)

This repository contains the code for DPO of LLM-jp models.

## Requirements

See `pyproject.toml` for the required packages.


## Installation

```bash
poetry install
```

## Training

Here is the command to train a model using 8 GPUs.

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch --config_file accelerate_configs/zero2.yaml train.py
```
