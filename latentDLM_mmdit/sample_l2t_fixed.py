# File: latentDLM_mmdit/sample_l2t_fixed.py
import torch
import torch.nn.functional as F
from tqdm import tqdm
import argparse
from pathlib import Path
import json
import sys
import os
import numpy as np
import yaml

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from train_mmdit import ContinuousDiffusion
from latentDLM_mmdit.models.multimodal_mmdit import MultimodalMMDiT
from latentDLM_mmdit.modeling_mmdit import get_tokenizer

class FixedL2TSampler:
    def __init__(self, checkpoint_path, config_path=None, device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.device = device
        print(f"Loading checkpoint from: {checkpoint_path}")
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        # Load config
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif 'config' in checkpoint and checkpoint['config'] is not None:
            config = checkpoint['config']
        else:
            config = {
                'model': {
                    'hidden_size': 1024,
                    'n_blocks': 24,
                    'n_heads': 24,
                    'cond_dim': 1024,
                    'max_seq_len': 1024,
                    'dropout': 0.1,
                    'num_residual_streams': 2,
                    'qk_rmsnorm': True,
                    'use_multimodal': True,
                    'latent_dim': 1024,
                }
            }
        
        # Get tokenizer
        self.tokenizer = get_tokenizer(config)
        self.mask_token_id = self.tokenizer.mask_token_id
        self.tokenizer_vocab_size = len(self.tokenizer)
        print(f"Tokenizer vocab size: {self.tokenizer_vocab_size}")
        
        # 关键：使用模型配置的词汇量
        model_vocab_size = config['model'].get('vocab_size', 30592)
        print(f"Model vocab size: {model_vocab_size}")
        
        if model_vocab_size != self.tokenizer_vocab_size:
            print(f"Warning: Model vocab size ({model_vocab_size}) != Tokenizer vocab size ({self.tokenizer_vocab_size})")
            print(f"Difference: {model_vocab_size - self.tokenizer_vocab_size} tokens")
        
        # Create model - 使用模型配置的词汇量
        latent_dim = config['model'].get('latent_dim', 1024)
        print(f"Creating model with latent_dim={latent_dim}, vocab_size={model_vocab_size}")
        
        self.model = MultimodalMMDiT(
            config=config['model'],
            vocab_size=model_vocab_size,  # 使用模型词汇量
            latent_dim=latent_dim,
            cluster_size=0
        ).to(device)
        
        # Load weights
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('model.'):
                    new_state_dict[k[6:]] = v
                else:
                    new_state_dict[k] = v
            
            missing, unexpected = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"Model loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        self.model.eval()
        self.model_vocab_size = model_vocab_size
    
    @torch.no_grad()
    def generate(self, latents, seq_len=128, steps=30, temperature=1.0):
        """Generate text from latents"""
        batch_size = latents.shape[0]
        latents = latents.to(self.device)
        
        # Initialize text with all masks
        text_tokens = torch.full((batch_size, seq_len), self.mask_token_id,
                                device=self.device, dtype=torch.long)
        
        text_timesteps = torch.linspace(1.0, 0.0, steps + 1, device=self.device)[:-1]
        latent_timesteps = torch.zeros(batch_size, device=self.device)
        
        for i in tqdm(range(steps), desc="Generating"):
            text_t = text_timesteps[i].expand(batch_size)
            
            # Forward pass
            text_logits, _ = self.model(
                text_tokens=text_tokens,
                latents=latents.unsqueeze(1),
                text_timesteps=text_t,
                latent_timesteps=latent_timesteps,
                attention_mask=None,
            )
            
            # 调试：检查形状
            if i == 0:
                print(f"\ntext_logits shape: {text_logits.shape}")
                print(f"Expected: [batch={batch_size}, seq={seq_len}, vocab={self.model_vocab_size}]")
            
            # 确保形状正确 [batch, seq, vocab]
            if text_logits.shape[-1] != self.model_vocab_size:
                print(f"Warning: Last dim {text_logits.shape[-1]} != model vocab {self.model_vocab_size}")
                # 尝试调整
                if text_logits.shape[1] == self.model_vocab_size:
                    text_logits = text_logits.transpose(1, 2)
                    print(f"Transposed to: {text_logits.shape}")
            
            # 截断到tokenizer的词汇量（如果模型词汇量更大）
            if text_logits.shape[-1] > self.tokenizer_vocab_size:
                print(f"Truncating from {text_logits.shape[-1]} to {self.tokenizer_vocab_size}")
                text_logits = text_logits[..., :self.tokenizer_vocab_size]
            
            # 应用温度
            if temperature != 1.0:
                text_logits = text_logits / temperature
            
            # 采样 - 使用截断后的词汇量
            probs = F.softmax(text_logits, dim=-1)
            probs_flat = probs.reshape(-1, probs.shape[-1])
            sampled_flat = torch.multinomial(probs_flat, 1)
            sampled = sampled_flat.view(batch_size, seq_len)
            
            # 更新掩码位置
            mask = (text_tokens == self.mask_token_id)
            if mask.any():
                text_tokens[mask] = sampled[mask]
            
            # 显示进度
            if i % 5 == 0:
                mask_ratio = mask.float().mean().item()
                print(f"Step {i+1}/{steps}: mask_ratio={mask_ratio:.3f}")
        
        return text_tokens
    
    def decode(self, tokens):
        """Decode tokens to text - 处理超出词汇表的token"""
        texts = []
        for t in tokens.cpu().numpy():
            valid = []
            for tok in t:
                # 检查token是否在tokenizer的词汇表中
                if tok >= self.tokenizer_vocab_size:
                    print(f"Warning: Token {tok} is out of vocabulary (vocab_size={self.tokenizer_vocab_size})")
                    continue
                if tok in [self.tokenizer.pad_token_id, 
                          getattr(self.tokenizer, 'cls_token_id', -1),
                          getattr(self.tokenizer, 'sep_token_id', -1),
                          self.mask_token_id]:
                    continue
                valid.append(tok)
            
            if valid:
                text = self.tokenizer.decode(valid, skip_special_tokens=True).strip()
                texts.append(text)
            else:
                texts.append("")
        return texts
    
    def load_latents(self, npy_dir, num_samples=None):
        """Load .npy files"""
        import glob
        files = sorted(glob.glob(os.path.join(npy_dir, "*.npy")))
        
        if num_samples and len(files) > num_samples:
            import random
            files = random.sample(files, num_samples)
        elif num_samples:
            files = files[:num_samples]
        
        latents = []
        for f in tqdm(files, desc="Loading latents"):
            data = np.load(f)
            latents.append(torch.from_numpy(data).float())
        
        if latents:
            return torch.stack(latents, dim=0)
        return torch.randn(num_samples or 3, 1024)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default=None)
    parser.add_argument("--npy_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output_dir", default="./l2t_fixed_output")
    
    args = parser.parse_args()
    
    # Create sampler
    sampler = FixedL2TSampler(args.checkpoint, args.config)
    
    # Load latents
    latents = sampler.load_latents(args.npy_dir, args.num_samples)
    print(f"Loaded {latents.shape[0]} latents")
    
    # Generate
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    all_texts = []
    all_tokens = []
    
    for i in range(0, latents.shape[0], args.batch_size):
        batch = latents[i:i+args.batch_size]
        print(f"\nGenerating batch {i//args.batch_size + 1}/{(latents.shape[0] + args.batch_size - 1)//args.batch_size}")
        
        tokens = sampler.generate(batch, steps=args.steps, temperature=args.temperature)
        texts = sampler.decode(tokens)
        
        all_texts.extend(texts)
        all_tokens.append(tokens.cpu())
        
        for j, text in enumerate(texts):
            idx = i + j
            print(f"\nSample {idx + 1}:")
            print(f"Text: {text}")
    
    # Save
    if all_tokens:
        all_tokens = torch.cat(all_tokens, dim=0)
    
    with open(output_dir / "results.json", "w", encoding='utf-8') as f:
        json.dump({
            'texts': all_texts,
            'num_samples': len(all_texts),
            'parameters': vars(args)
        }, f, ensure_ascii=False, indent=2)
    
    with open(output_dir / "texts.txt", "w", encoding='utf-8') as f:
        for i, text in enumerate(all_texts):
            f.write(f"Sample {i+1}:\n{text}\n\n")
    
    torch.save(latents, output_dir / "latents.pt")
    if all_tokens is not None:
        torch.save(all_tokens, output_dir / "tokens.pt")
    
    print(f"\nSaved {len(all_texts)} samples to {output_dir}")
    print(f"\nGenerated texts:")
    for i, text in enumerate(all_texts):
        print(f"{i+1}. {text}")

if __name__ == "__main__":
    main()