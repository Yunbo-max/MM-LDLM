# # File: latentDLM_mmdit/train_mmdit.py

# import datetime
# import json
# import os
# import random
# import sys
# import time
# from contextlib import contextmanager
# from pathlib import Path

# import hydra
# import numpy as np
# import torch
# import torch.distributed as dist
# import torch.nn as nn
# import torch.nn.functional as F
# import tqdm
# import wandb
# from omegaconf import OmegaConf, open_dict
# from torch.nn.parallel import DistributedDataParallel as DDP

# # Allow running from repo root or subdirs
# sys.path.append("..")
# sys.path.append(".")

# # Import MMDiT components
# from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
# from latentDLM_mmdit.checkpoints_mmdit import (
#     save_checkpoint,
#     load_checkpoint_for_training,
#     TrainingState,
#     save_rng_state,
#     load_rng_state,
# )
# from latentDLM_mmdit.modeling_mmdit import get_tokenizer
# from latentDLM_mmdit.optimizer import get_optimizer
# from latentDLM_mmdit.utils import (
#     get_lr,
#     parse_dtype,
#     calculate_flops_per_batch,
# )
# from latentDLM_mmdit.data_simple import get_simple_dataloaders
# from latentDLM_mmdit.diffusion_process import MaskedDiffusion


# class Logger:
#     def __init__(self, is_main_process: bool):
#         self.is_main_process = is_main_process

#     def init(self, *args, **kwargs):
#         if self.is_main_process:
#             wandb.init(*args, **kwargs)

#     def log(self, *args, **kwargs):
#         if self.is_main_process:
#             wandb.log(*args, **kwargs)


# def safe_barrier(local_rank: int | None = None) -> None:
#     """Call dist.barrier() only when the default process group is initialized.

#     Newer PyTorch versions may accept `device_ids` to disambiguate CUDA device.
#     """
#     if not (dist.is_available() and dist.is_initialized()):
#         return
#     try:
#         if local_rank is not None:
#             dist.barrier(device_ids=[local_rank])  # type: ignore[arg-type]
#         else:
#             dist.barrier()
#     except TypeError:
#         # Older PyTorch does not accept device_ids
#         dist.barrier()


# @contextmanager
# def main_process_first(local_rank: int | None = None):
#     """Context manager: rank0 runs the enclosed code first, others wait."""
#     if dist.is_available() and dist.is_initialized():
#         if dist.get_rank() == 0:
#             yield
#             safe_barrier(local_rank)
#         else:
#             safe_barrier(local_rank)
#             yield
#     else:
#         yield


# class ContinuousDiffusion:
#     """Simple continuous diffusion for latent vectors."""

#     def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02):
#         self.beta_min = beta_min
#         self.beta_max = beta_max

#     def get_alpha_beta(self, t: torch.Tensor):
#         """Get alpha and beta for continuous diffusion."""
#         beta = self.beta_min + (self.beta_max - self.beta_min) * t
#         alpha = 1 - beta
#         alpha_bar = torch.exp(torch.cumsum(torch.log(alpha), dim=0))
#         return alpha, beta, alpha_bar

#     def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
#         """Add noise to latents (continuous diffusion)."""
#         if noise is None:
#             noise = torch.randn_like(x0)

#         alpha_bar = self.get_alpha_beta(t)[2]
#         sqrt_alpha_bar = torch.sqrt(alpha_bar)
#         sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

#         # Reshape for broadcasting
#         if sqrt_alpha_bar.dim() == 1:
#             # Accept both [B,D] and [B,1,D]
#             if x0.dim() == 2:
#                 sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1)
#                 sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1)
#             elif x0.dim() == 3:
#                 sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1)
#                 sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1)

#         xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
#         return xt, noise

#     def sample_timesteps(self, batch_size: int, device: torch.device):
#         """Sample timesteps for continuous diffusion."""
#         return torch.rand(batch_size, device=device)


# class MultimodalDiffusionTrainer(nn.Module):
#     """Trainer for MMDiT with both text and latent diffusion."""

#     def __init__(self, model, tokenizer, text_noise_schedule, latent_diffusion, dtype: torch.dtype):
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.text_noise_schedule = text_noise_schedule
#         self.latent_diffusion = latent_diffusion
#         self.dtype = dtype

#         self.mask_token_id = tokenizer.mask_token_id

#     def forward(self, batch, force_transitting: bool = False):
#         # Extract data
#         input_ids = batch["input_ids"]
#         attention_mask = batch.get("attention_mask", None)
#         latents = batch.get("latent", None)

#         batch_size = input_ids.shape[0]
#         device = input_ids.device

#         # Sample timesteps for text diffusion
#         text_t = torch.rand(batch_size, device=device)
#         text_sigma = text_t.to(dtype=self.dtype)

#         # Apply text diffusion (masked diffusion)
#         noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_sigma)
#         text_target = input_ids
#         text_mask = (noisy_input_ids == self.mask_token_id)

#         # Handle latent diffusion
#         latent_t = None
#         noisy_latents = None
#         latent_target = None

#         if latents is not None:
#             latents = latents.to(device=device, dtype=self.dtype)

#             # Reduce possible sequence dimension -> [B, D]
#             if latents.dim() == 3:
#                 if latents.shape[1] == 1:
#                     latents = latents.squeeze(1)
#                 else:
#                     latents = latents.mean(dim=1)

#             # Sample timesteps for latent diffusion
#             latent_t = self.latent_diffusion.sample_timesteps(batch_size, device=device)
#             latent_t = latent_t.to(dtype=self.dtype)

#             # Add noise to latents
#             noise = torch.randn_like(latents)
#             noisy_latents, latent_noise = self.latent_diffusion.add_noise(latents, latent_t, noise)
#             latent_target = latent_noise

#             # Make shapes compatible with common MMDiT APIs ([B, 1, D])
#             if noisy_latents is not None and noisy_latents.dim() == 2:
#                 noisy_latents = noisy_latents.unsqueeze(1)
#             if latent_target is not None and latent_target.dim() == 2:
#                 latent_target = latent_target.unsqueeze(1)

#         # Forward pass through MMDiT
#         if latent_t is None:
#             latent_t = torch.zeros(batch_size, device=device, dtype=self.dtype)

#         # Convert attention_mask to boolean if needed
#         if attention_mask is not None and attention_mask.dtype != torch.bool:
#             attention_mask = attention_mask.bool()

#         text_logits, latent_pred = self.model(
#             text_tokens=noisy_input_ids,
#             latents=noisy_latents,
#             text_timesteps=text_sigma,
#             latent_timesteps=latent_t,
#             attention_mask=attention_mask,
#         )

#         vocab_size = text_logits.shape[-1]

#         # Text loss (denoising objective)
#         text_loss = F.cross_entropy(
#             text_logits.reshape(-1, vocab_size),
#             text_target.reshape(-1),
#             ignore_index=-100,
#         )

#         # Latent loss (MSE on noise prediction) - only if latents exist
#         latent_loss = torch.tensor(0.0, device=device, dtype=text_loss.dtype)
#         if latent_pred is not None and latent_target is not None:
#             # Avoid silent broadcasting: align singleton dims if present
#             if latent_pred.dim() == 3 and latent_target.dim() == 2:
#                 latent_target = latent_target.unsqueeze(1)
#             if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
#                 latent_target = latent_target.squeeze(1)
#             if latent_pred.shape != latent_target.shape:
#                 raise ValueError(
#                     f"latent_pred/latent_target shape mismatch: {latent_pred.shape} vs {latent_target.shape}"
#                 )
#             latent_loss = F.mse_loss(latent_pred, latent_target)

#         total_loss = text_loss + latent_loss

#         # Compute metrics
#         with torch.no_grad():
#             pred_tokens = torch.argmax(text_logits, dim=-1)
#             if text_mask.any():
#                 text_accuracy = (pred_tokens[text_mask] == text_target[text_mask]).float().mean().item()
#             else:
#                 text_accuracy = 0.0

#             metrics = {
#                 "loss": float(total_loss.item()),
#                 "text_loss": float(text_loss.item()),
#                 "latent_loss": float(latent_loss.item()),
#                 "text_accuracy": float(text_accuracy),
#             }

#             if latents is not None:
#                 metrics["latent_norm"] = float(latents.norm(dim=-1).mean().item())
#                 if latent_pred is not None:
#                     metrics["latent_pred_norm"] = float(latent_pred.norm(dim=-1).mean().item())

#         return total_loss, metrics


# def _init_distributed() -> tuple[int, int, int, bool, bool]:
#     """Initialize distributed training if launched with torchrun.

#     Returns:
#         local_rank, global_rank, world_size, is_main_process, is_distributed
#     """
#     env_has_ddp = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
#     if not env_has_ddp:
#         return 0, 0, 1, True, False

#     if not torch.cuda.is_available():
#         raise RuntimeError("Distributed launch detected (torchrun), but CUDA is not available.")

#     local_rank = int(os.environ["LOCAL_RANK"])
    
#     # Safety check: ensure the GPU device exists before setting it
#     num_gpus = torch.cuda.device_count()
#     if local_rank >= num_gpus:
#         raise RuntimeError(
#             f"LOCAL_RANK={local_rank} but only {num_gpus} GPU(s) available. "
#             f"Please ensure --nproc_per_node <= {num_gpus} and CUDA_VISIBLE_DEVICES is set correctly."
#         )
    
#     torch.cuda.set_device(local_rank)

#     init_kwargs = dict(
#         backend="nccl",
#         timeout=datetime.timedelta(minutes=30),
#         init_method="env://",
#     )

#     try:
#         # Newer PyTorch supports device_id to set PG default device.
#         dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", local_rank))  # type: ignore[arg-type]
#     except TypeError:
#         dist.init_process_group(**init_kwargs)
#     except Exception as e:
#         raise RuntimeError(
#             "Failed to initialize torch.distributed process group. "
#             f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
#             f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
#         ) from e

#     global_rank = dist.get_rank()
#     world_size = dist.get_world_size()
#     is_main_process = (global_rank == 0)
#     is_distributed = dist.is_available() and dist.is_initialized()
#     return local_rank, global_rank, world_size, is_main_process, is_distributed


# @hydra.main(config_path="configs", config_name="mmdit", version_base="1.1")
# def main(config):
#     # ---------------- Distributed init ----------------
#     local_rank, global_rank, world_size, is_main_process, is_distributed = _init_distributed()

#     with open_dict(config):
#         config.training.world_size = world_size
#         config.training.local_rank = local_rank
#         config.training.global_rank = global_rank

#     # ---------------- Seeding ----------------
#     seed = config.training.seed + global_rank
#     torch.manual_seed(seed)
#     random.seed(seed)
#     np.random.seed(seed)

#     # ---------------- CUDA perf knobs ----------------
#     torch.set_float32_matmul_precision("high")
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
#     try:
#         torch.backends.cuda.enable_flash_sdp(True)  # PyTorch 2.1+
#     except Exception:
#         pass

#     dtype = parse_dtype(config.training.dtype)
#     if torch.cuda.is_available():
#         device = torch.device("cuda", local_rank)
#     else:
#         device = torch.device("cpu")

#     print(f"Using device={device} and dtype={dtype}")

#     # ---------------- Build / load model + trainer ----------------
#     if config.training.resume is None:
#         tokenizer = get_tokenizer(config)
#         vocab_size = len(tokenizer)

#         model = MultimodalMMDiT(
#             config=config.model,
#             vocab_size=vocab_size,
#             latent_dim=config.model.get("latent_dim", 768),
#             cluster_size=config.model.get("cluster_size", 0),
#         ).to(device=device, dtype=dtype)

#         text_noise_schedule = MaskedDiffusion(tokenizer)

#         latent_diffusion = ContinuousDiffusion(
#             beta_min=config.model.get("latent_beta_min", 0.0001),
#             beta_max=config.model.get("latent_beta_max", 0.02),
#         )

#         trainer = MultimodalDiffusionTrainer(
#             model=model,
#             tokenizer=tokenizer,
#             text_noise_schedule=text_noise_schedule,
#             latent_diffusion=latent_diffusion,
#             dtype=dtype,
#         ).to(device=device)

#         optimizer = get_optimizer(config, trainer)

#         state = TrainingState(
#             epoch=0,
#             epoch_start_step=0,
#             step=0,
#         )
#     else:
#         (
#             model,
#             text_noise_schedule,
#             tokenizer,
#             old_config,
#             trainer,
#             optimizer,
#             state,
#         ) = load_checkpoint_for_training(config.training.resume, device=device, dtype=dtype)

#     # ---------------- Data ----------------
#     with main_process_first(local_rank):
#         train_dl, test_dl = get_simple_dataloaders(config, tokenizer)

#     max_lr = config.optimizer.lr

#     # ---------------- Logging (W&B) ----------------
#     logger = Logger(is_main_process)
#     # Respect external WANDB_DIR if set; otherwise use config
#     os.environ.setdefault("WANDB_DIR", config.logging.get("wandb_dir", "./outputs/"))
#     logger.init(
#         name=config.logging.run_name,
#         entity=config.logging.wandb_entity,
#         project=config.logging.wandb_project,
#         config=OmegaConf.to_container(config, resolve),
#     )

#     if is_main_process:
#         pwd = Path(".").resolve()
#         wandb.config.update({"pwd": pwd})
#         print(f"Working directory: {pwd}")

#     non_emb_params = sum(p.numel() for p in model.mmdit.parameters())
#     flops_per_batch = calculate_flops_per_batch(
#         config, model, len(tokenizer), non_emb_params, method="hoffmann"
#     )
#     trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

#     # ---------------- Optional compilation ----------------
#     if config.training.compile_model:
#         try:
#             opt_trainer = torch.compile(trainer)
#         except RuntimeError as e:
#             if "Python 3.13" in str(e):
#                 print("Warning: torch.compile not supported on Python 3.13+, skipping compilation")
#                 opt_trainer = trainer
#             else:
#                 raise
#     else:
#         opt_trainer = trainer

#     # ---------------- DDP wrap ----------------
#     if is_distributed:
#         ddp_trainer = DDP(opt_trainer, device_ids=[local_rank], output_device=local_rank)
#     else:
#         ddp_trainer = opt_trainer

#     if is_main_process:
#         non_emb_params_str = (
#             f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
#         )
#         trainable_params_str = (
#             f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
#         )
#         print("*** Starting MMDiT with Joint Text-Latent Diffusion ***")
#         print(f"* World size: {world_size}")
#         print(f"* FLOPs per batch: {flops_per_batch:.3g}")
#         print(f"* Per-device batch size: {config.training.train_batch_size}")
#         print(f"* Total batch size: {config.training.train_batch_size * world_size}")
#         print(f"* Non-embedding parameters: {non_emb_params_str}")
#         print(f"* Trainable parameters: {trainable_params_str}")
#         print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
#         print(f"* Latent dimension: {config.model.get('latent_dim', 768)}")
#         print("* Text diffusion: Masked Diffusion")
#         print(
#             f"* Latent diffusion: Continuous Diffusion "
#             f"(beta={config.model.get('latent_beta_min', 0.0001)}-{config.model.get('latent_beta_max', 0.02)})"
#         )
#         print("*************************")

#     # ---------------- Training loop setup ----------------
#     if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
#         train_dl.sampler.set_epoch(state.epoch)
#     batch_iterator = iter(train_dl)

#     # Initialize eval dataloader
#     _ = next(iter(test_dl))

#     if state.step - state.epoch_start_step > 0:
#         for _ in tqdm.trange(
#             state.step - state.epoch_start_step,
#             desc="Skipping batches",
#             dynamic_ncols=True,
#             disable=not is_main_process,
#         ):
#             next(batch_iterator)

#     curr_time = time.time()
#     trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)
#     state.start_time = curr_time - trained_time
#     state.curr_time = curr_time
#     prev_time = curr_time

#     log_buffer = []

#     if config.training.resume is not None:
#         load_rng_state(config.training.resume, global_rank)

#     # Gradient accumulation (effective batch size = train_batch_size * grad_accum_steps)
#     grad_accum_steps = max(1, int(getattr(config.training, "grad_accum_steps", 1)))
#     optimizer.zero_grad(set_to_none=True)

#     # Initialize loss logging
#     loss_log_file = None
#     if is_main_process:
#         log_dir = Path(config.logging.save_dir) / config.logging.run_name
#         log_dir.mkdir(parents=True, exist_ok=True)
#         loss_log_file = log_dir / "training_log.jsonl"
#         loss_log_file.write_text("")

#     # ---------------- Train ----------------
#     with tqdm.tqdm(
#         total=config.training.num_train_steps,
#         initial=state.step,
#         desc="Training",
#         ncols=100,
#         disable=not is_main_process,
#         leave=True,
#     ) as pbar:
#         for step in range(state.step, config.training.num_train_steps):
#             try:
#                 batch = next(batch_iterator)
#             except StopIteration:
#                 state.epoch += 1
#                 state.epoch_start_step = step
#                 if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
#                     train_dl.sampler.set_epoch(state.epoch)
#                 batch_iterator = iter(train_dl)
#                 batch = next(batch_iterator)

#             curr_lr = get_lr(config, max_lr, step)
#             for param_group in optimizer.param_groups:
#                 param_group["lr"] = curr_lr

#             batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

#             loss, metrics = ddp_trainer(batch)

#             # Single-line progress update with key metrics
#             if step % 10 == 0 and is_main_process:
#                 pbar.set_postfix(
#                     {
#                         "Loss": f"{loss.item():.4f}",
#                         "Text": f"{metrics.get('text_loss', 0.0):.4f}",
#                         "Latent": f"{metrics.get('latent_loss', 0.0):.4f}",
#                         "Acc": f"{metrics.get('text_accuracy', 0.0):.4f}",
#                     }
#                 )

#             # Log all losses to file (every step)
#             if is_main_process and loss_log_file is not None:
#                 log_entry = {
#                     "step": int(step),
#                     "loss": float(loss.item()),
#                     "lr": float(curr_lr),
#                 }
#                 for key, value in metrics.items():
#                     if isinstance(value, (int, float)):
#                         log_entry[key] = float(value)
#                     elif hasattr(value, "item"):
#                         log_entry[key] = float(value.item())
#                 with open(loss_log_file, "a") as f:
#                     f.write(json.dumps(log_entry) + "\n")

#             # Scale loss for accumulation
#             scaled_loss = (loss * config.loss.loss_scale) / grad_accum_steps
#             scaled_loss.backward()

#             norm_value = float("nan")
#             should_step = ((step + 1) % grad_accum_steps == 0) or ((step + 1) == config.training.num_train_steps)
#             if should_step:
#                 # Grad clip
#                 if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
#                     norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
#                 else:
#                     norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1e6)
#                 norm_value = float(norm.item())

#                 if torch.isnan(norm):
#                     print(f"Warning: NaN gradient detected at step {step}")
#                     for param in trainer.parameters():
#                         if param.grad is not None:
#                             param.grad.data.zero_()

#                 optimizer.step()
#                 optimizer.zero_grad(set_to_none=True)

#             # Logging throughput
#             batch_tokens = (
#                 batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
#                 .sum()
#                 .item()
#                 * config.training.world_size
#             )
#             batch_flops = flops_per_batch * config.training.world_size
#             total_batch_size = batch["input_ids"].size(0) * config.training.world_size

#             state.total_tokens += batch_tokens
#             state.total_flops += batch_flops

#             curr_time = time.time()
#             step_time = curr_time - prev_time
#             prev_time = curr_time

#             log_buffer.append(
#                 {
#                     "train/loss": float(loss.item()),
#                     "train/lr": float(curr_lr),
#                     "train/step": int(step + 1),
#                     "train/grad_norm": norm_value,
#                     "train/epoch": float(step / len(train_dl)),
#                     "train/total_tokens": float(state.total_tokens),
#                     "train/total_flops": float(state.total_flops),
#                     "train/tokens_per_sec": float(batch_tokens / step_time),
#                     "train/flops_per_sec": float(batch_flops / step_time),
#                     "train/samples_per_sec": float(total_batch_size / step_time),
#                     "train/it_per_sec": float(1.0 / step_time),
#                     "train/avg_it_per_sec": float((step + 1) / (curr_time - state.start_time)),
#                     **{f"train/{k}": float(v) for k, v in metrics.items()},
#                 }
#             )

#             if ((step + 1) % config.logging.log_freq) == 0:
#                 avg_metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
#                 logger.log(avg_metrics, step=step)
#                 logger.log({"trainer/global_step": step}, step=step)
#                 log_buffer = []

#             # ---------------- Evaluation ----------------
#             if ((step + 1) % config.logging.eval_freq) == 0:
#                 with torch.no_grad():
#                     eval_start_time = time.time()
#                     ddp_trainer.eval()

#                     eval_metrics = {}
#                     eval_loss = 0.0
#                     num_eval_samples = 0

#                     for i, test_batch in enumerate(
#                         tqdm.tqdm(
#                             test_dl,
#                             desc="Eval",
#                             dynamic_ncols=True,
#                             total=config.logging.num_eval_batches,
#                             disable=not is_main_process,
#                         )
#                     ):
#                         bs = test_batch["input_ids"].size(0)

#                         test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
#                         e_loss, e_metrics = ddp_trainer(test_batch)

#                         for k, v in e_metrics.items():
#                             eval_metrics[k] = eval_metrics.get(k, 0.0) + float(v) * bs

#                         eval_loss += float(e_loss.item()) * bs
#                         num_eval_samples += bs

#                         if i >= config.logging.num_eval_batches - 1:
#                             break

#                     eval_elapsed_time = time.time() - eval_start_time

#                     if is_main_process and num_eval_samples > 0:
#                         logger.log(
#                             {
#                                 "eval/loss": eval_loss / num_eval_samples,
#                                 "eval/time_taken": eval_elapsed_time,
#                                 **{f"eval/{k}": v / num_eval_samples for k, v in eval_metrics.items()},
#                             },
#                             step=step,
#                         )

#                     ddp_trainer.train()

#             # ---------------- Save checkpoint ----------------
#             state.step += 1
#             if ((step + 1) % config.logging.save_freq) == 0:
#                 output_path = Path(config.logging.save_dir, config.logging.run_name)
#                 suffix = "latest"
#                 if (step + 1) == 500000:
#                     suffix = "-500k"
#                 elif (step + 1) == 1000000:
#                     suffix = "-1M"
#                 elif (step + 1) == 250000:
#                     suffix = "-250k"
#                 output_path = output_path / suffix

#                 # Ensure directory exists (rank0), then sync.
#                 if is_main_process:
#                     output_path.mkdir(exist_ok=True, parents=True)
#                 safe_barrier(local_rank)

#                 # Save checkpoint on rank0 only.
#                 if is_main_process:
#                     save_checkpoint(output_path, trainer, optimizer, state)

#                 # Sync to ensure checkpoint is fully written.
#                 safe_barrier(local_rank)

#                 # Save RNG state per-rank (requires directory to exist).
#                 save_rng_state(output_path, global_rank)

#                 # Final sync (optional).
#                 safe_barrier(local_rank)

#             pbar.update(1)

#     if is_distributed:
#         dist.destroy_process_group()


# if __name__ == "__main__":
#     main()



# File: latentDLM_mmdit/train_mmdit.py

import datetime
import json
import os
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import wandb
from omegaconf import OmegaConf, open_dict
from torch.nn.parallel import DistributedDataParallel as DDP

# Allow running from repo root or subdirs
sys.path.append("..")
sys.path.append(".")
os.environ["WANDB_MODE"] = "disabled"

# Import MMDiT components
from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
from latentDLM_mmdit.checkpoints_mmdit import (
    save_checkpoint,
    load_checkpoint_for_training,
    TrainingState,
    save_rng_state,
    load_rng_state,
)
from latentDLM_mmdit.modeling_mmdit import get_tokenizer
from latentDLM_mmdit.optimizer import get_optimizer
from latentDLM_mmdit.utils import (
    get_lr,
    parse_dtype,
    calculate_flops_per_batch,
)
from latentDLM_mmdit.data_simple import get_simple_dataloaders
from latentDLM_mmdit.diffusion_process import MaskedDiffusion


class Logger:
    def __init__(self, is_main_process: bool):
        self.is_main_process = is_main_process

    def init(self, *args, **kwargs):
        if self.is_main_process:
            wandb.init(*args, **kwargs)

    def log(self, *args, **kwargs):
        if self.is_main_process:
            wandb.log(*args, **kwargs)


def safe_barrier(local_rank: int | None = None) -> None:
    """Call dist.barrier() only when the default process group is initialized.

    Newer PyTorch versions may accept `device_ids` to disambiguate CUDA device.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    try:
        if local_rank is not None:
            dist.barrier(device_ids=[local_rank])  # type: ignore[arg-type]
        else:
            dist.barrier()
    except TypeError:
        # Older PyTorch does not accept device_ids
        dist.barrier()


@contextmanager
def main_process_first(local_rank: int | None = None):
    """Context manager: rank0 runs the enclosed code first, others wait."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            yield
            safe_barrier(local_rank)
        else:
            safe_barrier(local_rank)
            yield
    else:
        yield


class ContinuousDiffusion:
    """Simple continuous diffusion for latent vectors."""

    def __init__(self, beta_min: float = 0.0001, beta_max: float = 0.02):
        self.beta_min = beta_min
        self.beta_max = beta_max

    def get_alpha_beta(self, t: torch.Tensor):
        """Get alpha and beta for continuous diffusion."""
        beta = self.beta_min + (self.beta_max - self.beta_min) * t
        alpha = 1 - beta
        alpha_bar = torch.exp(torch.cumsum(torch.log(alpha), dim=0))
        return alpha, beta, alpha_bar

    def add_noise(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None):
        """Add noise to latents (continuous diffusion)."""
        if noise is None:
            noise = torch.randn_like(x0)

        alpha_bar = self.get_alpha_beta(t)[2]
        sqrt_alpha_bar = torch.sqrt(alpha_bar)
        sqrt_one_minus_alpha_bar = torch.sqrt(1 - alpha_bar)

        # Reshape for broadcasting
        if sqrt_alpha_bar.dim() == 1:
            # Accept both [B,D] and [B,1,D]
            if x0.dim() == 2:
                sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1)
                sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1)
            elif x0.dim() == 3:
                sqrt_alpha_bar = sqrt_alpha_bar.view(-1, 1, 1)
                sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.view(-1, 1, 1)

        xt = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise
        return xt, noise

    def sample_timesteps(self, batch_size: int, device: torch.device):
        """Sample timesteps for continuous diffusion."""
        return torch.rand(batch_size, device=device)


# class MultimodalDiffusionTrainer(nn.Module):
#     """Trainer for MMDiT with both text and latent diffusion."""

#     def __init__(self, model, tokenizer, text_noise_schedule, latent_diffusion, dtype: torch.dtype):
#         super().__init__()
#         self.model = model
#         self.tokenizer = tokenizer
#         self.text_noise_schedule = text_noise_schedule
#         self.latent_diffusion = latent_diffusion
#         self.dtype = dtype

#         self.mask_token_id = tokenizer.mask_token_id

#     def forward(self, batch, force_transitting: bool = False):
#         # Extract data
#         input_ids = batch["input_ids"]
#         attention_mask = batch.get("attention_mask", None)
#         latents = batch.get("latent", None)

#         batch_size = input_ids.shape[0]
#         device = input_ids.device

#         # Sample timesteps for text diffusion
#         text_t = torch.rand(batch_size, device=device)
#         text_sigma = text_t.to(dtype=self.dtype)

#         # Apply text diffusion (masked diffusion)
#         noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_sigma)
#         text_target = input_ids
#         text_mask = (noisy_input_ids == self.mask_token_id)

#         # Handle latent diffusion
#         latent_t = None
#         noisy_latents = None
#         latent_target = None

#         if latents is not None:
#             latents = latents.to(device=device, dtype=self.dtype)

#             # Reduce possible sequence dimension -> [B, D]
#             if latents.dim() == 3:
#                 if latents.shape[1] == 1:
#                     latents = latents.squeeze(1)
#                 else:
#                     latents = latents.mean(dim=1)

#             # Sample timesteps for latent diffusion
#             latent_t = self.latent_diffusion.sample_timesteps(batch_size, device=device)
#             latent_t = latent_t.to(dtype=self.dtype)

#             # Add noise to latents
#             noise = torch.randn_like(latents)
#             noisy_latents, latent_noise = self.latent_diffusion.add_noise(latents, latent_t, noise)
#             latent_target = latent_noise

#             # Make shapes compatible with common MMDiT APIs ([B, 1, D])
#             if noisy_latents is not None and noisy_latents.dim() == 2:
#                 noisy_latents = noisy_latents.unsqueeze(1)
#             if latent_target is not None and latent_target.dim() == 2:
#                 latent_target = latent_target.unsqueeze(1)

#         # Forward pass through MMDiT
#         if latent_t is None:
#             latent_t = torch.zeros(batch_size, device=device, dtype=self.dtype)

#         # Convert attention_mask to boolean if needed
#         if attention_mask is not None and attention_mask.dtype != torch.bool:
#             attention_mask = attention_mask.bool()

#         text_logits, latent_pred = self.model(
#             text_tokens=noisy_input_ids,
#             latents=noisy_latents,
#             text_timesteps=text_sigma,
#             latent_timesteps=latent_t,
#             attention_mask=attention_mask,
#         )

#         vocab_size = text_logits.shape[-1]

#         # Text loss (denoising objective)
#         text_loss = F.cross_entropy(
#             text_logits.reshape(-1, vocab_size),
#             text_target.reshape(-1),
#             ignore_index=-100,
#         ).to(dtype=text_logits.dtype)

#         # Latent loss (MSE on noise prediction) - only if latents exist
#         latent_loss = torch.tensor(0.0, device=device, dtype=text_loss.dtype)
#         if latent_pred is not None and latent_target is not None:
#             # Avoid silent broadcasting: align singleton dims if present
#             if latent_pred.dim() == 3 and latent_target.dim() == 2:
#                 latent_target = latent_target.unsqueeze(1)
#             if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
#                 latent_target = latent_target.squeeze(1)
#             if latent_pred.shape != latent_target.shape:
#                 raise ValueError(
#                     f"latent_pred/latent_target shape mismatch: {latent_pred.shape} vs {latent_target.shape}"
#                 )
#             latent_loss = F.mse_loss(latent_pred, latent_target)

#         total_loss = text_loss + latent_loss

#         # Compute metrics
#         with torch.no_grad():
#             pred_tokens = torch.argmax(text_logits, dim=-1)
#             if text_mask.any():
#                 text_accuracy = (pred_tokens[text_mask] == text_target[text_mask]).float().mean().item()
#             else:
#                 text_accuracy = 0.0

#             metrics = {
#                 "loss": float(total_loss.item()),
#                 "text_loss": float(text_loss.item()),
#                 "latent_loss": float(latent_loss.item()),
#                 "text_accuracy": float(text_accuracy),
#             }

#             if latents is not None:
#                 metrics["latent_norm"] = float(latents.norm(dim=-1).mean().item())
#                 if latent_pred is not None:
#                     metrics["latent_pred_norm"] = float(latent_pred.norm(dim=-1).mean().item())

#         return total_loss, metrics


class MultimodalDiffusionTrainer(nn.Module):
    def __init__(self, model, tokenizer, text_noise_schedule, latent_diffusion, 
                 dtype: torch.dtype, config):
        
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.text_noise_schedule = text_noise_schedule
        self.latent_diffusion = latent_diffusion
        self.dtype = dtype
        self.config = config
        
        self.mask_token_id = tokenizer.mask_token_id
        
        # Loss type configuration
        self.loss_type = config.training.loss_type
#         self.loss_weights = getattr(config.training, "loss_type_weights", {})
        
        # For sequential training
        self.current_epoch = 0
        self.sequential_schedule = getattr(config.training, "sequential_schedule", [])
    
    def get_training_mode(self, batch_idx=None):
        """Determine which training mode to use based on config."""
        
        if self.loss_type == "sequential" and self.sequential_schedule:
            # Sequential training by epoch
            total_epochs = sum(s["epochs"] for s in self.sequential_schedule)
            current_epoch = self.current_epoch % total_epochs
            
            cumulative = 0
            for schedule in self.sequential_schedule:
                cumulative += schedule["epochs"]
                if current_epoch < cumulative:
                    return schedule["type"]
            
            # Fallback
            return "mixed"
        
        # elif self.loss_type == "mixed":
        #     # Weighted random sampling based on config weights
        #     modes = ["unconditional", "l2t", "t2l", "partial"]
        #     weights = [
        #         self.loss_weights.get("unconditional", 0.25),
        #         self.loss_weights.get("l2t", 0.25),
        #         self.loss_weights.get("t2l", 0.25),
        #         self.loss_weights.get("partial", 0.25)
        #     ]
        #     # Normalize weights
        #     weights = torch.tensor(weights) / sum(weights)
        #     return np.random.choice(modes, p=weights.numpy())
        
        # elif self.loss_type == "alternating":
        #     # Alternate between modes every N batches
        #     if batch_idx is None:
        #         batch_idx = 0
        #     cycle_length = getattr(self.config.training, "alternating_cycle", 100)
        #     mode_idx = (batch_idx // cycle_length) % 4
        #     modes = ["unconditional", "l2t", "t2l", "partial"]
        #     return modes[mode_idx]
        
        else:
            # Fixed mode: "unconditional", "l2t", or "t2l"
            return self.loss_type
    
    def forward(self, batch, batch_idx=None, force_transitting: bool = False):
        # Get current training mode
        mode = self.get_training_mode(batch_idx)
        
        # Extract data
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)
        latents = batch.get("latent", None)

        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Initialize variables
        text_t = None
        latent_t = None
        noisy_input_ids = None
        noisy_latents = None
        text_target = None
        text_mask = None
        latent_target = None
        
        # Text timestep sampling
        def sample_text_t():
            return torch.rand(batch_size, device=device)
        
        # Latent timestep sampling
        def sample_latent_t():
            return self.latent_diffusion.sample_timesteps(batch_size, device=device)
        
        # ===== MODE: Unconditional/Joint Generation =====
        if mode == "unconditional":
            # Both modalities get noise
            text_t = sample_text_t()
            latent_t = sample_latent_t() if latents is not None else None
            
            noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_t)
            text_target = input_ids
            text_mask = (noisy_input_ids == self.mask_token_id)
            
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )
            
            text_loss_weight = 1.0
            latent_loss_weight = 1.0
            
        # ===== MODE: Latent → Text =====
        elif mode == "l2t":
            # Text: fully masked (generate from latents)
            text_t = torch.ones(batch_size, device=device)  # Full noise
            noisy_input_ids = torch.full_like(input_ids, self.mask_token_id)
            text_target = input_ids
            text_mask = torch.ones_like(text_target, dtype=torch.bool)  # All masked
            
            # Latent: clean (conditioning)
            latent_t = torch.zeros(batch_size, device=device) if latents is not None else None
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noisy_latents = latents  # Clean latent as input
                latent_target = torch.zeros_like(latents)  # Zero target for MSE
            
            text_loss_weight = 1.0
            latent_loss_weight = 0.0  # No latent loss for this mode
            
        # ===== MODE: Text → Latent =====
        elif mode == "t2l":
            # Text: clean (conditioning)
            text_t = torch.zeros(batch_size, device=device)
            noisy_input_ids = input_ids  # Clean text
            text_target = input_ids
            text_mask = torch.zeros_like(text_target, dtype=torch.bool)  # No masking
            
            # Latent: fully noisy (generate from text)
            latent_t = torch.ones(batch_size, device=device) if latents is not None else None
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )
            
            text_loss_weight = 0.0  # No text loss for this mode
            latent_loss_weight = 1.0
            
        # ===== MODE: Partial Conditioning =====
        elif mode == "partial":
            # Both partially noisy (some information from each)
            text_t = torch.rand(batch_size, device=device) * 0.5  # Up to 50% masking
            latent_t = torch.rand(batch_size, device=device) * 0.5 if latents is not None else None
            
            noisy_input_ids = self.text_noise_schedule.sample_zt(input_ids, text_t)
            text_target = input_ids
            text_mask = (noisy_input_ids == self.mask_token_id)
            
            if latents is not None:
                latents = latents.to(device=device, dtype=self.dtype)
                if latents.dim() == 3 and latents.shape[1] == 1:
                    latents = latents.squeeze(1)
                noise = torch.randn_like(latents)
                noisy_latents, latent_target = self.latent_diffusion.add_noise(
                    latents, latent_t, noise
                )
            
            text_loss_weight = 1.0
            latent_loss_weight = 1.0
        
        # ===== FORWARD PASS =====
        text_sigma = text_t.to(dtype=self.dtype)
        if latent_t is not None:
            latent_t = latent_t.to(dtype=self.dtype)
        
        # Handle attention mask
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = attention_mask.bool()
        
        # Forward pass
        text_logits, latent_pred = self.model(
            text_tokens=noisy_input_ids,
            latents=noisy_latents.unsqueeze(1) if noisy_latents is not None else None,
            text_timesteps=text_sigma,
            latent_timesteps=latent_t,
            attention_mask=attention_mask,
        )
        
        vocab_size = text_logits.shape[-1]
        
        # ===== LOSS CALCULATION =====
        # Text loss
        if text_loss_weight > 0:
            if text_mask is not None and text_mask.any():
                text_loss_unmasked = F.cross_entropy(
                    text_logits.view(-1, vocab_size),
                    text_target.view(-1),
                    ignore_index=-100,
                    reduction='none'
                )
                text_loss = (text_loss_unmasked * text_mask.view(-1)).sum() / text_mask.sum().clamp(min=1)
            else:
                text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        else:
            text_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # Latent loss
        if (latent_loss_weight > 0 and latent_pred is not None and 
            latent_target is not None and not torch.all(latent_target == 0)):
            # Ensure shapes match
            if latent_pred.dim() == 3 and latent_target.dim() == 2:
                latent_target = latent_target.unsqueeze(1)
            if latent_pred.dim() == 2 and latent_target.dim() == 3 and latent_target.shape[1] == 1:
                latent_target = latent_target.squeeze(1)
            
            # Handle shape mismatches
            if latent_pred.shape != latent_target.shape:
                if latent_pred.dim() == 3 and latent_pred.shape[1] > 1:
                    latent_pred = latent_pred.mean(dim=1)
                if latent_target.dim() == 3 and latent_target.shape[1] > 1:
                    latent_target = latent_target.mean(dim=1)
            
            latent_loss = F.mse_loss(latent_pred, latent_target)
        else:
            latent_loss = torch.tensor(0.0, device=device, dtype=self.dtype)
        
        # Weighted total loss
        total_loss = text_loss_weight * text_loss + latent_loss_weight * latent_loss
        
        # ===== METRICS =====
        with torch.no_grad():
            pred_tokens = torch.argmax(text_logits, dim=-1)
            if text_mask is not None and text_mask.any():
                text_accuracy = (pred_tokens[text_mask] == text_target[text_mask]).float().mean().item()
            else:
                text_accuracy = 0.0
            
            metrics = {
                "loss": float(total_loss.item()),
                "text_loss": float(text_loss.item()),
                "latent_loss": float(latent_loss.item()),
                "text_accuracy": float(text_accuracy),
                # "mode": mode,  # Track which mode was used
            }
            
            if latents is not None:
                metrics["latent_norm"] = float(latents.norm(dim=-1).mean().item())
                if latent_pred is not None:
                    metrics["latent_pred_norm"] = float(latent_pred.norm(dim=-1).mean().item())
        
        return total_loss, metrics

def _init_distributed() -> tuple[int, int, int, bool, bool]:
    """Initialize distributed training if launched with torchrun.

    Returns:
        local_rank, global_rank, world_size, is_main_process, is_distributed
    """
    env_has_ddp = all(k in os.environ for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    if not env_has_ddp:
        return 0, 0, 1, True, False

    if not torch.cuda.is_available():
        raise RuntimeError("Distributed launch detected (torchrun), but CUDA is not available.")

    local_rank = int(os.environ["LOCAL_RANK"])
    
    # Safety check: ensure the GPU device exists before setting it
    num_gpus = torch.cuda.device_count()
    if local_rank >= num_gpus:
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} but only {num_gpus} GPU(s) available. "
            f"Please ensure --nproc_per_node <= {num_gpus} and CUDA_VISIBLE_DEVICES is set correctly."
        )
    
    torch.cuda.set_device(local_rank)

    init_kwargs = dict(
        backend="nccl",
        timeout=datetime.timedelta(minutes=30),
        init_method="env://",
    )

    try:
        # Newer PyTorch supports device_id to set PG default device.
        dist.init_process_group(**init_kwargs, device_id=torch.device("cuda", local_rank))  # type: ignore[arg-type]
    except TypeError:
        dist.init_process_group(**init_kwargs)
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize torch.distributed process group. "
            f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}, "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')}"
        ) from e

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    is_main_process = (global_rank == 0)
    is_distributed = dist.is_available() and dist.is_initialized()
    return local_rank, global_rank, world_size, is_main_process, is_distributed


@hydra.main(config_path="configs", config_name="mmdit", version_base="1.1")
def main(config):
    # ---------------- Distributed init ----------------
    local_rank, global_rank, world_size, is_main_process, is_distributed = _init_distributed()

    with open_dict(config):
        config.training.world_size = world_size
        config.training.local_rank = local_rank
        config.training.global_rank = global_rank

    # ---------------- Seeding ----------------
    seed = config.training.seed + global_rank
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # ---------------- CUDA perf knobs ----------------
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # BF16-specific optimizations (if available)
    if hasattr(torch.backends.cuda, 'matmul') and hasattr(torch.backends.cuda.matmul, 'allow_bf16_reduced_precision_reduction'):
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    if hasattr(torch.backends.cudnn, 'allow_bf16_reduced_precision_reduction'):
        torch.backends.cudnn.allow_bf16_reduced_precision_reduction = True
    print("BF16 optimizations enabled")
        
    try:
        torch.backends.cuda.enable_flash_sdp(True)  # PyTorch 2.1+
    except Exception:
        pass

    dtype = parse_dtype(config.training.dtype)
    if torch.cuda.is_available():
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    print(f"Using device={device} and dtype={dtype}")

    # ---------------- Build / load model + trainer ----------------
    if config.training.resume is None:
        tokenizer = get_tokenizer(config)
        vocab_size = len(tokenizer)

        model = MultimodalMMDiT(
            config=config.model,
            vocab_size=vocab_size,
            latent_dim=config.model.get("latent_dim", 768),
            cluster_size=config.model.get("cluster_size", 0),
        ).to(device=device, dtype=dtype)
        
        

        text_noise_schedule = MaskedDiffusion(tokenizer)

        latent_diffusion = ContinuousDiffusion(
            beta_min=config.model.get("latent_beta_min", 0.0001),
            beta_max=config.model.get("latent_beta_max", 0.02),
        )

        trainer = MultimodalDiffusionTrainer(
            model=model,
            tokenizer=tokenizer,
            text_noise_schedule=text_noise_schedule,
            latent_diffusion=latent_diffusion,
            dtype=dtype,
            config=config
        ).to(device=device)

        optimizer = get_optimizer(config, trainer)

        state = TrainingState(
            epoch=0,
            epoch_start_step=0,
            step=0,
        )
    else:
        (
            model,
            text_noise_schedule,
            tokenizer,
            old_config,
            trainer,
            optimizer,
            state,
        ) = load_checkpoint_for_training(config.training.resume, device=device, dtype=dtype)

    # ---------------- Data ----------------
    with main_process_first(local_rank):
        train_dl, test_dl = get_simple_dataloaders(config, tokenizer)

    max_lr = config.optimizer.lr

    # ---------------- Logging (W&B) ----------------
    logger = Logger(is_main_process)
    # Respect external WANDB_DIR if set; otherwise use config
    os.environ.setdefault("WANDB_DIR", config.logging.get("wandb_dir", "./outputs/"))
    logger.init(
        name=config.logging.run_name,
        entity=config.logging.wandb_entity,
        project=config.logging.wandb_project,
        config=OmegaConf.to_container(config, resolve=True),
    )

    if is_main_process:
        pwd = Path(".").resolve()
        wandb.config.update({"pwd": pwd})
        print(f"Working directory: {pwd}")

    non_emb_params = sum(p.numel() for p in model.mmdit.parameters())
    flops_per_batch = calculate_flops_per_batch(
        config, model, len(tokenizer), non_emb_params, method="hoffmann"
    )
    trainable_params = sum(p.numel() for p in trainer.parameters() if p.requires_grad)

    # ---------------- Optional compilation ----------------
    if config.training.compile_model:
        try:
            opt_trainer = torch.compile(trainer)
        except RuntimeError as e:
            if "Python 3.13" in str(e):
                print("Warning: torch.compile not supported on Python 3.13+, skipping compilation")
                opt_trainer = trainer
            else:
                raise
    else:
        opt_trainer = trainer

    # ---------------- DDP wrap ----------------
    if is_distributed:
        ddp_trainer = DDP(opt_trainer, device_ids=[local_rank], output_device=local_rank,find_unused_parameters=True)
    else:
        ddp_trainer = opt_trainer

    if is_main_process:
        non_emb_params_str = (
            f"{non_emb_params / 1e6:.1f}M" if non_emb_params < 500 * 1e6 else f"{non_emb_params / 1e9:.1f}B"
        )
        trainable_params_str = (
            f"{trainable_params / 1e6:.1f}M" if trainable_params < 500 * 1e6 else f"{trainable_params / 1e9:.1f}B"
        )
        print("*** Starting MMDiT with Joint Text-Latent Diffusion ***")
        print(f"* World size: {world_size}")
        print(f"* FLOPs per batch: {flops_per_batch:.3g}")
        print(f"* Per-device batch size: {config.training.train_batch_size}")
        print(f"* Total batch size: {config.training.train_batch_size * world_size}")
        print(f"* Non-embedding parameters: {non_emb_params_str}")
        print(f"* Trainable parameters: {trainable_params_str}")
        print(f"* Model dtype: {next(iter(model.parameters())).dtype}")
        print(f"* Latent dimension: {config.model.get('latent_dim', 768)}")
        print("* Text diffusion: Masked Diffusion")
        print(
            f"* Latent diffusion: Continuous Diffusion "
            f"(beta={config.model.get('latent_beta_min', 0.0001)}-{config.model.get('latent_beta_max', 0.02)})"
        )
        print("*************************")

    # ---------------- Training loop setup ----------------
    if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
        train_dl.sampler.set_epoch(state.epoch)
    batch_iterator = iter(train_dl)

    # Initialize eval dataloader
    _ = next(iter(test_dl))

    if state.step - state.epoch_start_step > 0:
        for _ in tqdm.trange(
            state.step - state.epoch_start_step,
            desc="Skipping batches",
            dynamic_ncols=True,
            disable=not is_main_process,
        ):
            next(batch_iterator)

    curr_time = time.time()
    trained_time = 0 if config.training.resume is None else (state.start_time - state.curr_time)
    state.start_time = curr_time - trained_time
    state.curr_time = curr_time
    prev_time = curr_time

    log_buffer = []

    if config.training.resume is not None:
        load_rng_state(config.training.resume, global_rank)

    # Initialize loss logging
    loss_log_file = None
    if is_main_process:
        log_dir = Path(config.logging.save_dir) / config.logging.run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        loss_log_file = log_dir / "training_log.jsonl"
        loss_log_file.write_text("")

    # ---------------- Train ----------------
    with tqdm.tqdm(
        total=config.training.num_train_steps,
        initial=state.step,
        desc="Training",
        ncols=100,
        disable=not is_main_process,
        leave=True,
    ) as pbar:
        for step in range(state.step, config.training.num_train_steps):
            try:
                batch = next(batch_iterator)
            except StopIteration:
                state.epoch += 1
                state.epoch_start_step = step
                if is_distributed and hasattr(train_dl.sampler, "set_epoch"):
                    train_dl.sampler.set_epoch(state.epoch)
                batch_iterator = iter(train_dl)
                batch = next(batch_iterator)

            curr_lr = get_lr(config, max_lr, step)
            for param_group in optimizer.param_groups:
                param_group["lr"] = curr_lr

            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}

            loss, metrics = ddp_trainer(batch)

            # Single-line progress update with key metrics
            if step % 10 == 0 and is_main_process:
                pbar.set_postfix(
                    {
                        "Loss": f"{loss.item():.4f}",
                        "Text": f"{metrics.get('text_loss', 0.0):.4f}",
                        "Latent": f"{metrics.get('latent_loss', 0.0):.4f}",
                        "Acc": f"{metrics.get('text_accuracy', 0.0):.4f}",
                    }
                )

            # Log all losses to file (every step)
            if is_main_process and loss_log_file is not None:
                log_entry = {
                    "step": int(step),
                    "loss": float(loss.item()),
                    "lr": float(curr_lr),
                }
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        log_entry[key] = float(value)
                    elif hasattr(value, "item"):
                        log_entry[key] = float(value.item())
                with open(loss_log_file, "a") as f:
                    f.write(json.dumps(log_entry) + "\n")

            (loss * config.loss.loss_scale).backward()

            # Grad clip
            if config.optimizer.grad_clip_norm and config.optimizer.grad_clip_norm > 0:
                norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), config.optimizer.grad_clip_norm)
            else:
                norm = torch.nn.utils.clip_grad_norm_(trainer.parameters(), 1e6)

            if torch.isnan(norm):
                print(f"Warning: NaN gradient detected at step {step}")
                for param in trainer.parameters():
                    if param.grad is not None:
                        param.grad.data.zero_()

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Logging throughput
            batch_tokens = (
                batch.get("attention_mask", torch.ones_like(batch["input_ids"]))
                .sum()
                .item()
                * config.training.world_size
            )
            batch_flops = flops_per_batch * config.training.world_size
            total_batch_size = batch["input_ids"].size(0) * config.training.world_size

            state.total_tokens += batch_tokens
            state.total_flops += batch_flops

            curr_time = time.time()
            step_time = curr_time - prev_time
            prev_time = curr_time

            log_buffer.append(
                {
                    "train/loss": float(loss.item()),
                    "train/lr": float(curr_lr),
                    "train/step": int(step + 1),
                    "train/grad_norm": float(norm.item()),
                    "train/epoch": float(step / len(train_dl)),
                    "train/total_tokens": float(state.total_tokens),
                    "train/total_flops": float(state.total_flops),
                    "train/tokens_per_sec": float(batch_tokens / step_time),
                    "train/flops_per_sec": float(batch_flops / step_time),
                    "train/samples_per_sec": float(total_batch_size / step_time),
                    "train/it_per_sec": float(1.0 / step_time),
                    "train/avg_it_per_sec": float((step + 1) / (curr_time - state.start_time))
                  
                }
            )

            if ((step + 1) % config.logging.log_freq) == 0:
                avg_metrics = {k: sum(d[k] for d in log_buffer) / len(log_buffer) for k in log_buffer[0]}
                logger.log(avg_metrics, step=step)
                logger.log({"trainer/global_step": step}, step=step)
                log_buffer = []

            # ---------------- Evaluation ----------------
                    
            if ((step + 1) % config.logging.eval_freq) == 0:
                with torch.no_grad():
                    eval_start_time = time.time()
                    ddp_trainer.eval()

                    eval_metrics = {}
                    eval_loss = 0.0
                    num_eval_samples = 0

                    for i, test_batch in enumerate(
                        tqdm.tqdm(
                            test_dl,
                            desc="Eval",
                            dynamic_ncols=True,
                            total=config.logging.num_eval_batches,
                            disable=not is_main_process,
                        )
                    ):
                        bs = test_batch["input_ids"].size(0)

                        test_batch = {k: v.to(device, non_blocking=True) for k, v in test_batch.items()}
                        e_loss, e_metrics = ddp_trainer(test_batch)

                        # FIX THIS: Only accumulate numeric metrics
                        for k, v in e_metrics.items():
                            try:
                                # Try to convert to float
                                eval_metrics[k] = eval_metrics.get(k, 0.0) + float(v) * bs
                            except (ValueError, TypeError):
                                # Skip non-numeric metrics
                                pass

                        eval_loss += float(e_loss.item()) * bs
                        num_eval_samples += bs

                        if i >= config.logging.num_eval_batches - 1:
                            break

                    eval_elapsed_time = time.time() - eval_start_time

                    # Re-enable this logging if you want eval logging
                    if is_main_process and num_eval_samples > 0:
                        logger.log(
                            {
                                "eval/loss": eval_loss / num_eval_samples,
                                "eval/time_taken": eval_elapsed_time,
                                **{f"eval/{k}": v / num_eval_samples for k, v in eval_metrics.items()},
                            },
                            step=step,
                        )

                    ddp_trainer.train()

            # ---------------- Save checkpoint ----------------
            state.step += 1
            if ((step + 1) % config.logging.save_freq) == 0:
                output_path = Path(config.logging.save_dir, config.logging.run_name)
                suffix = "latest"
                if (step + 1) == 500000:
                    suffix = "-500k"
                elif (step + 1) == 1000000:
                    suffix = "-1M"
                elif (step + 1) == 250000:
                    suffix = "-250k"
                output_path = output_path / suffix

                # Ensure directory exists (rank0), then sync.
                if is_main_process:
                    output_path.mkdir(exist_ok=True, parents=True)
                safe_barrier(local_rank)

                # Save checkpoint on rank0 only.
                if is_main_process:
                    save_checkpoint(output_path, trainer, optimizer, state)

                # Sync to ensure checkpoint is fully written.
                safe_barrier(local_rank)

                # Save RNG state per-rank (requires directory to exist).
                save_rng_state(output_path, global_rank)

                # Final sync (optional).
                safe_barrier(local_rank)

            pbar.update(1)

    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()