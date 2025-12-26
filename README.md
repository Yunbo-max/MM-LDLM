# MM-LDLM — Multimodal Latent Diffusion Language Models

This repository contains code for latent reasoning and multimodal MMDiT models (text ↔ latent, image ↔ latent) and accompanying preprocessing and baseline code. The README here focuses on the latent/text/image MMDiT workflows and how to preprocess data, run encoders, and start training the latent models. Baseline code (HDLM, MDLM, AR, GIDD+) is preserved in the `baseline` and `baseline_latent` folders.

This repo now contains (high-level):

- `baseline/` and `baseline_latent/` — baseline reproductions (HDLM, MDLM, AR, GIDD+, and other baselines).
- `latentDLM_mmdit/` — latent MMDiT (text-to-latent / latent reasoning) training and utilities (train_mmdit.py, trainer_multimodal.py, etc.).
- `latentIMG_mmdit/` — image+text MMDiT training scripts and image patch encoders (continuous & discrete training variants).
- `preprocessed_data/` — data preprocessing helpers and distributed latent extraction: `prepare_data_multi_gpu.py` and usage examples in its README.
- `baseline_latent/` — additional latent-focused baseline code (train_latent_dit.py, trainer_latent.py, etc.).

Note: This top-level README intentionally focuses on the latent/text/image workflows. If you need to run the original HDLM unconditional generation or NeurIPS experiment reproductions, see the `baseline/` and `baseline_latent/` folders — those baselines are retained.



---

## Pip install

Create a Python virtual environment and install requirements:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# optionally: pip install -e .
```

---

## Data format and preprocessing

Preprocess / extract text latents (if you plan to train continuous latent models): follow `preprocessed_data/README` (it contains example `torchrun` commands using `prepare_data_multi_gpu.py`).

Example preprocessing (from `preprocessed_data/README`):

```bash
# SONAR on 2 GPUs
torchrun --nnodes=1 --nproc_per_node=1 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model sonar \
  --batch-size 128 \
  --max-samples 10000000 \
  --output-dir preprocessed_data/sonar_1024d_full

# E5 (multilingual-e5-large) on 4 GPUs
torchrun --nnodes=1 --nproc_per_node=1 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model e5 \
  --batch-size 256 \
  --max-samples 10000000 \
  --output-dir preprocessed_data/e5_1024d_full

# Qwen embedding model on 2 GPUs (lower memory than large causal models)
torchrun --nnodes=1 --nproc_per_node=1 preprocessed_data/prepare_data_multi_gpu.py \
  --datasets openwebtext \
  --latent-model qwen \
  --batch-size 8 \
  --max-samples 10000000 \
  --output-dir preprocessed_data/qwen_1024d_full
```

Files created by `prepare_data_multi_gpu.py` (typical):

- output-dir/
  - texts/train/*.txt
  - latents/train/*.npy
  - train_data.json
  - validation_data.json (optional)

Make sure embedding dimensionality from the encoder matches the `--dim-text` / model config used by training scripts. The `preprocessed_data/README` has more details and checks.


---

## Baselines

The repo preserves baseline implementations. See:

- `baseline/` — HDLM (original hierarchical diffusion language model) and other baselines.
- `baseline_latent/` — latent-focused baselines for DIT-style latent training.

You can run baseline training via the scripts in those folders. The original README content for the baselines has been preserved there; consult the respective `README` or `configs/` files in each baseline folder for exact training commands.

### Example baseline run commands

Below are example single-node, single-process `torchrun` commands for common baseline runs (copied from the provided commands, with minor fixes and `training.dtype=bf16` applied to the `mmdit` example). Adjust paths, `--nnodes` and `--nproc_per_node` for your hardware.

```bash
# MDLM baseline (quick test)
torchrun --nnodes 1 --nproc_per_node 1 baseline/train.py \
  --config-name mdlm \
  logging.run_name="'test-openwebtext'" \
  data.dataset_name="openwebtext" \
  data.dataset_subset=null \
  data.data_files.train=null \
  data.cache_dir="./data" \
  training.train_batch_size=4 \
  training.eval_batch_size=4 \
  logging.eval_freq=10 \
  logging.log_freq=5 \
  training.compile_model=False
```

```bash
# Latent DIT baseline (full latent training)
torchrun --nnodes 1 --nproc_per_node 1 baseline_latent/train_latent_dit.py \
  --config-name mdlm_latent \
  logging.run_name="latent-full-8M" \
  data.dataset_name="openwebtext" \
  data.latent_data_root="/inspire/hdd/global_user/zhangjiaquan-253108540222//latent/HDLM/mmdit/data_root" \
  model.latent_dim=768 \
  model.use_latent_conditioning=true \
  training.train_batch_size=32 \
  training.eval_batch_size=32 \
  training.num_train_steps=250000 \
  training.compile_model=false 
```

```bash
# Cross-attention latent baseline
torchrun --nnodes 1 --nproc_per_node 1 baseline_latent/train_cross_dit.py \
  --config-name mdlm_cross_attention \
  logging.run_name="cross-attention-training" \
  data.dataset_name="openwebtext" \
  data.latent_data_root="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/HDLM/mmdit/data_root" \
  training.train_batch_size=16 \
  training.eval_batch_size=16 \
  training.num_train_steps=250000 \
  training.compile_model=false
```

---

### Training latent MMDiT

# MMDIT (text ↔ latent)

```bash
# MMDiT training run (use bf16 for faster/memory-efficient runs on supported hardware)
torchrun --nnodes 1 --nproc_per_node 1 latentDLM_mmdit/train_mmdit.py \
  --config-name mmdit \
  logging.run_name="mmdit-training-h200" \
  training.train_batch_size=16 \
  training.eval_batch_size=16 \
  training.num_train_steps=25000 \
  training.compile_model=false \
  model.latent_dim=1024 \
  training.dtype=bf16
```



# MMDIT  (text ↔ Image)
```bash
python latentIMG_mmdit/train_image_continuous.py \
  --data-root /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/LatentMAS/data/coco2014/images \
  --epochs 50 \
  --batch-size 8 \
  --dim-text 1024 \
  --dim-image 512 \
  --output-dir outputs_image_continuous
```

```bash
python latentIMG_mmdit/train_image_discrete.py \
  --data-root /inspire/hdd/global_user/zhangjiaquan-253108540222/latent/LatentMAS/data/coco2014/images \
  --epochs 50 \
  --batch-size 8 \
  --dim-text 768 \
  --dim-image 512 \
  --output-dir outputs_image_masked
```

For multi-GPU runs, wrap the training script with `torchrun` and ensure the training scripts initialize DDP correctly (the preprocessing script already supports distributed extraction).

---

## Notes and tips

- Always verify embedding dimension when switching encoders: mismatch between encoder embedding size and `--dim-text` will break training.
- For quick pipeline checks, set `--max-samples 10` to validate model loading, tokenizer/embedding loading, and I/O.
- If you plan to use Qwen or BGE models, ensure your environment has the necessary packages (device_map handling, FlagEmbedding if using BGE wrapper).



### Evaluation

There are also a couple of scripts to run inference and evaluate the trained models.

#### Generate samples
The following command will generate `num_samples=256` samples in `num_denoising_steps=512` iterations from the model checkpoint located at `path` and save them to `samples_dir=samples.pt`.
```bash
python hdlm/eval/generate_samples.py path=./outputs/path/to/checkpoint/ samples_dir=samples.pt num_samples=256 num_denoising_steps=512 batch_size=16
```

#### Generative PPL
Given a file containing samples generated with the `generate_samples.py` script, the following command will compute the generative PPL.
Here we assume that the diffusion model used to generate samples located at `samples.pt` uses the `gpt2` tokenizer, and we compute generative PPL using `gpt2-large` as a reference model.
The results will be saved to `metrics_path=metrics.json`.
```bash
python hdlm/eval/generative_ppl.py samples_path=samples.pt model_tokenizer=gpt2 pretrained_model=gpt2-large batch_size=1 metrics_path=metrics.json
```

#### Validation loss
A simple helper script to compute the loss of a trained model on the entire validation split.
```bash
python hdlm/eval/loss.py path=./outputs/path/to/checkpoint/ batch_size=32
```



## 💞 Acknowledgements
The code is built upon the below repositories, we thank all the contributors for open-sourcing.
* [GIDD](https://github.com/dvruette/gidd/)
* [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
