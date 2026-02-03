# PEFT Training Scripts - Usage Guide

This directory contains scalable scripts for training visual language models with PEFT (Parameter-Efficient Fine-Tuning).

## Files Overview

- **create_dataset.py**: Prepares the dataset for training
- **create_peft_wrapper.py**: Tests the PEFT wrapper with a specific model
- **train_model.py**: Main training script with full PEFT configuration
- **trainer_callback.py**: Custom callback for logging training metrics
- **get_linear_layers.py**: Utility to inspect model linear layers

The `*_copy.py` files are the original versions before refactoring.

## 1. create_dataset.py

Prepares the dataset by loading data from JSON, filtering by map, and creating HuggingFace dataset format.

### Usage

```bash
python create_dataset.py \
  --data-json-path ./data_updated.json \
  --training-map Town01 \
  --output-path /path/to/output/dataset \
  --prompt-template /path/to/prompt.txt  # Optional
```

### Arguments

- `--data-json-path`: Path to the JSON file containing the data (default: `./data_updated.json`)
- `--training-map`: CARLA map name to filter (default: `Town01`)
- `--output-path`: Where to save the processed dataset (default: `/datafast/105-1/Datasets/INTERNS/aplanaj/hf_dataset_bev_ss_coord`)
- `--prompt-template`: Optional path to a text file with custom prompt template

### Example

```bash
# Basic usage with defaults
python create_dataset.py

# Custom configuration
python create_dataset.py \
  --data-json-path /data/my_data.json \
  --training-map Town02 \
  --output-path /data/my_dataset
```

## 2. create_peft_wrapper.py

Tests the PEFT wrapper by loading a model and processing sample data. Useful for debugging.

### Usage

```bash
python create_peft_wrapper.py \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --dataset-path /path/to/dataset \
  --num-samples 1
```

### Arguments

- `--model-id`: HuggingFace model ID (default: `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`)
- `--dataset-path`: Path to the dataset (default: `/datafast/105-1/Datasets/INTERNS/aplanaj/hf_dataset_ss_coord`)
- `--num-samples`: Number of samples to test (default: `1`)

### Example

```bash
# Test with Qwen model
python create_peft_wrapper.py \
  --model-id Qwen/Qwen3-VL-32B-Instruct \
  --dataset-path /data/my_dataset \
  --num-samples 3
```

## 3. train_model.py

Main training script with comprehensive configuration options.

### Usage

```bash
python train_model.py \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --module language \
  --dataset-path /path/to/dataset \
  --output-dir /path/to/output \
  --num-epochs 3 \
  --batch-size 1 \
  --learning-rate 2e-4 \
  --lora-r 16 \
  --lora-alpha 32 \
  --use-dora \
  --cuda-visible-devices 0,1
```

### Arguments

#### Model Configuration
- `--model-id`: Model to use (choices: `Qwen/Qwen3-VL-32B-Instruct`, `google/gemma-3-12b-it`, `lmms-lab/LLaVA-OneVision-1.5-8B-Instruct`)
- `--module`: Which module to fine-tune (choices: `all-linear`, `vision`, `language`, `CNN`)

#### Paths
- `--dataset-path`: Path to the dataset
- `--output-dir`: Output directory (auto-generated if not specified)

#### Training Hyperparameters
- `--num-epochs`: Number of training epochs (default: `3`)
- `--batch-size`: Per-device training batch size (default: `1`)
- `--gradient-accumulation-steps`: Gradient accumulation steps (default: `4`)
- `--learning-rate`: Learning rate (default: `2e-4`)
- `--warmup-steps`: Number of warmup steps (default: `50`)
- `--logging-steps`: Logging frequency (default: `10`)

#### LoRA Configuration
- `--lora-r`: LoRA rank (default: `16`)
- `--lora-alpha`: LoRA alpha parameter (default: `32`)
- `--use-dora`: Use DoRA instead of LoRA (flag)

#### GPU Selection
- `--cuda-visible-devices`: Comma-separated GPU IDs (e.g., `0,1,2`)

### Examples

#### Train LLaVA language module with default settings
```bash
python train_model.py \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --module language \
  --dataset-path /data/my_dataset
```

#### Train Gemma vision module with custom LoRA settings
```bash
python train_model.py \
  --model-id google/gemma-3-12b-it \
  --module vision \
  --dataset-path /data/my_dataset \
  --lora-r 64 \
  --lora-alpha 128 \
  --use-dora \
  --num-epochs 5 \
  --cuda-visible-devices 0,1,2,3
```

#### Train Qwen with all-linear module
```bash
python train_model.py \
  --model-id Qwen/Qwen3-VL-32B-Instruct \
  --module all-linear \
  --dataset-path /data/my_dataset \
  --output-dir /output/qwen-all-linear \
  --learning-rate 1e-4 \
  --batch-size 2
```

## Complete Workflow Example

```bash
# Step 1: Create dataset
python create_dataset.py \
  --data-json-path ./data_updated.json \
  --training-map Town01 \
  --output-path /data/town01_dataset

# Step 2: (Optional) Test the wrapper
python create_peft_wrapper.py \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --dataset-path /data/town01_dataset \
  --num-samples 3

# Step 3: Train the model
python train_model.py \
  --model-id lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
  --module language \
  --dataset-path /data/town01_dataset \
  --output-dir /output/llava-language-town01 \
  --num-epochs 3 \
  --lora-r 16 \
  --lora-alpha 32 \
  --use-dora \
  --cuda-visible-devices 0,1
```

## Module Types Explained

- **all-linear**: Trains all linear layers in the model
- **vision**: Trains only the visual encoder and projector
- **language**: Trains only the language model (LLM)
- **CNN**: Trains only the CNN patch embedding layer

## Supported Models

1. **Qwen/Qwen3-VL-32B-Instruct**: Large vision-language model with 4-bit quantization
2. **google/gemma-3-12b-it**: Google's Gemma 3 multimodal model
3. **lmms-lab/LLaVA-OneVision-1.5-8B-Instruct**: LLaVA OneVision model

## Notes

- The scripts automatically handle model-specific tokenization and chat templates
- Training logs are saved to `training_logs.jsonl` in the output directory
- Models are saved with both weights and processor configuration
- All scripts support `--help` flag for detailed argument information
