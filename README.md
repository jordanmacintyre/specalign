# specalign

A PyTorch implementation comparing General Knowledge Distillation (GKD), DistillSpec, and a novel contrastive rank-order loss function for training efficient student models to align with teacher distributions.

## Overview

This repository explores different approaches to knowledge distillation for language models:

1. **General Knowledge Distillation (GKD)**: On-policy distillation where the student generates sequences that are then evaluated by both student and teacher models, with KL divergence used to align distributions.

2. **DistillSpec**: An alternative to GKD that improves speculative decoding by better aligning student and teacher output distributions.

3. **Contrastive Rank-Order Loss**: A novel loss function focusing on rank-order logit alignment rather than explicit distribution matching, designed to improve speculative decoding stability and acceptance rates.

## Key Features

- **On-policy generation**: Student model generates sequences during training for more realistic distribution alignment
- **Multiple distillation approaches**: GKD, DistillSpec comparison, and custom contrastive loss
- **Efficient training**: Gradient accumulation, mixed precision (bfloat16), and optional 4-bit quantization for teacher models
- **Configurable hyperparameters**: Temperature scaling, learning rates, generation parameters

## Installation

### Option 1: Using Conda (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/specalign.git
cd specalign

# Create and activate conda environment
conda env create -f environment.yml
conda activate specalign

# Install the package in editable mode
pip install -e .
```

### Option 2: Using pip only

```bash
# Clone the repository
git clone https://github.com/yourusername/specalign.git
cd specalign

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision

# Install the package with training dependencies
pip install -e ".[train]"
```

### Optional Dependencies

**Development tools** (linting, testing):
```bash
pip install -e ".[dev]"
```

**Flash Attention 2** (recommended for faster training):
```bash
pip install flash-attn --no-build-isolation
```

### Requirements

- **Python**: 3.12 (specified in pyproject.toml)
- **CUDA**: Compatible GPU for training (recommended)
- **Key dependencies**: PyTorch, Transformers (>=4.43), Datasets (>=2.20), Accelerate (>=0.33), BitsAndBytes (>=0.43)

## Usage

### Training with GKD

The main training script implements General Knowledge Distillation:

```bash
python src/specalign/main.py
```

This script supports decoder-only models:
- **Decoder-only**: Currently configured for Qwen models (0.5B student, 1.5B teacher)
- **Task**: Summarization on CNN/DailyMail dataset
- **On-policy generation**: Student generates K_MAX tokens per iteration (default: 4)
- **KL divergence loss**: Aligns student logits to teacher logits with temperature scaling

### Key Hyperparameters

**Training configuration** (in `src/specalign/main.py`):
- **student_id** / **teacher_id**: HuggingFace model identifiers (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
- **K_MAX**: Number of tokens student generates per iteration (default: 4)
- **KD_TEMP**: Temperature for KL divergence softening (default: 1.0)
- **LR**: Learning rate (2e-4)
- **GRAD_ACCUM_STEPS**: Gradient accumulation steps (16)
- **MAX_STEPS**: Total training steps (2,000)
- **MAX_INPUT_SEQ_LENGTH**: Maximum input sequence length (512)
- **device**: CUDA device to use (e.g., "cuda:1")
- **attn_implementation**: Attention implementation (e.g., "flash_attention_2")

## Evaluation Metrics

During training, the following metrics are tracked and logged every 50 steps:

- **kd**: KL divergence loss between student and teacher logit distributions (scaled by T²)
- **agree**: Agreement rate - fraction of tokens where argmax(student) == argmax(teacher)
- **k**: Number of tokens generated per on-policy generation step
- **dt**: Time elapsed (in seconds) since last log

Example training output:
```
[step 50] kd=2.341 agree=0.723 k=4 dt=45.2s
[step 100] kd=2.156 agree=0.751 k=4 dt=44.8s
[step 150] kd=2.089 agree=0.768 k=4 dt=43.9s
```

**Downstream evaluation**: Trained student models can be evaluated for:
- **Speculative decoding acceptance rate**: How often teacher accepts student proposals
- **Generation quality**: Comparison of outputs when using student vs. teacher alone
- **Inference speedup**: Latency improvements from speculative decoding

## Project Structure

```
specalign/
├── notebooks/              # Jupyter notebooks for experiments
│   ├── gkd.ipynb          # GKD experiments and analysis
│   ├── gkd_encdec.ipynb   # Encoder-decoder GKD experiments
│   ├── gkd_dev.ipynb      # Development/testing notebook
│   ├── sft.ipynb          # Supervised fine-tuning experiments
│   └── encdec_dev.ipynb   # Encoder-decoder development
├── src/specalign/          # Core training code
│   ├── main.py            # Main GKD training script
│   ├── utils.py           # On-policy generation utilities
│   └── __init__.py
├── Notes.md               # Research notes
└── README.md
```

## Implementation Details

### GKD Training Loop

1. **On-policy generation**: Student generates k tokens from prompts (k randomized per batch)
2. **Dual forward pass**: Both student and teacher process the complete sequence
3. **Logit extraction**: Extract logits for generated tokens only (using prompt length masks)
4. **KD loss computation**: KL divergence between temperature-scaled distributions
5. **Backpropagation**: Only student model is updated; teacher remains frozen

### Optimizations

- **Mixed precision**: bfloat16 training with autocast for memory efficiency
- **Gradient accumulation**: Effective batch size = `BATCH_SIZE × GRAD_ACCUM`
- **Teacher quantization**: Optional 4-bit quantization via BitsAndBytes
- **Flash Attention 2**: Faster attention computation when available
- **Cosine learning rate schedule**: Warmup + cosine decay for stable training

### Contrastive Loss (In Development)

The contrastive rank-order loss focuses on:
- Preserving relative token rankings rather than exact probability distributions
- Improved acceptance rates in speculative decoding scenarios
- Potentially more stable training dynamics

## Research Goals

This repository aims to:

1. **Compare GKD vs. DistillSpec**: Evaluate which distillation approach produces better student models for speculative decoding
2. **Develop contrastive loss**: Design and test a rank-order-based loss function as an alternative to KL divergence
3. **Measure downstream impact**: Assess how different training objectives affect speculative decoding acceptance rates and inference speed
4. **Support multiple architectures**: Explore both decoder-only (Gemma, Qwen) and encoder-decoder (FLAN-T5) models

## References

- [GKD: Generalized Knowledge Distillation for Auto-regressive Sequence Models](https://arxiv.org/abs/2306.13649) (Agarwal et al., 2023)
- [DistillSpec: Improving Speculative Decoding via Knowledge Distillation](https://arxiv.org/abs/2310.08461) (Zhou et al., 2023)
- [Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) (Leviathan et al., 2022)
- [Distilling the Knowledge in a Neural Network](https://arxiv.org/abs/1503.02531) (Hinton et al., 2015)

## License

MIT License - see LICENSE file for details