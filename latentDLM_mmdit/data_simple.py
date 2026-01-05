# # # File: latentDLM_mmdit/data_simple.py
# # import json
# # from pathlib import Path
# # import torch
# # import numpy as np
# # from datasets import Dataset
# # import random


# # File: latentDLM_mmdit/data_simple.py (CORRECTED)
# import json
# from pathlib import Path
# import torch
# import numpy as np
# from torch.utils.data import Dataset  # Import PyTorch's base Dataset
# import random


# class SimpleLatentDataset(Dataset):  # Inherit ONLY from torch.utils.data.Dataset
#     """Simple dataset that loads text and latent pairs from JSON."""
    
#     def __init__(self, json_path, tokenizer, max_length=512, max_samples=None):
#         self.json_path = Path(json_path)
#         self.tokenizer = tokenizer
#         self.max_length = max_length
        
#         # Load JSON data
#         with open(json_path, 'r') as f:
#             self.samples = json.load(f)  # Changed variable name from 'data' to 'samples'
        
#         if max_samples:
#             self.samples = self.samples[:max_samples]
        
#         print(f"Loaded {len(self.samples)} samples from {json_path}")
        
#     def __len__(self):
#         return len(self.samples)
    
#     def __getitem__(self, idx):
#         item = self.samples[idx]  # Use the new name 'samples' here
#         text = item.get('text', '')
        
#         # Tokenize text
#         tokenized = self.tokenizer(
#             text,
#             truncation=True,
#             padding='max_length',
#             max_length=self.max_length,
#             return_tensors='pt'
#         )
        
#         result = {
#             'input_ids': tokenized['input_ids'].squeeze(0),
#             'attention_mask': tokenized['attention_mask'].squeeze(0),
#         }
        
#         # Load latent if available
#         if 'latent_path' in item:
#             latent_path = Path(self.json_path.parent) / item['latent_path']
#             if latent_path.exists():
#                 try:
#                     latent = np.load(latent_path)
#                     # Ensure latent is a tensor of shape [1, latent_dim]
#                     latent_tensor = torch.from_numpy(latent).float()
#                     if latent_tensor.dim() == 1:
#                         latent_tensor = latent_tensor.unsqueeze(0)
#                     result['latent'] = latent_tensor
#                 except Exception as e:
#                     print(f"Warning: Could not load latent from {latent_path}: {e}")
#                     # Optionally, create a zero latent as fallback
#                     # latent_dim = config.model.get('latent_dim', 768)
#                     # result['latent'] = torch.zeros(1, latent_dim)
        
#         return result


# def get_simple_dataloaders(config, tokenizer):
#     """Simple dataloader that works with JSON files and latents."""
#     from torch.utils.data import DataLoader
#     from torch.utils.data.distributed import DistributedSampler
    
#     # Create datasets
#     train_json = Path(config.data.data_files.train)
#     val_json = Path(config.data.data_files.validation) if hasattr(config.data, 'data_files') and hasattr(config.data.data_files, 'validation') else None
    
#     train_ds = SimpleLatentDataset(
#         train_json,
#         tokenizer,
#         max_length=config.model.max_seq_len,
#         max_samples=config.data.get('max_samples', None)
#     )
    
#     if val_json and val_json.exists():
#         test_ds = SimpleLatentDataset(
#             val_json,
#             tokenizer,
#             max_length=config.model.max_seq_len,
#             max_samples=config.data.get('max_val_samples', 1000)
#         )
#     else:
#         # Create validation split from training
#         print("No validation JSON found, splitting training data")
#         total_size = len(train_ds)
#         val_size = min(1000, total_size // 10)
#         indices = list(range(total_size))
#         random.shuffle(indices)
#         val_indices = indices[:val_size]
#         train_indices = indices[val_size:]
        
#         from torch.utils.data import Subset
#         test_ds = Subset(train_ds, val_indices)
#         train_ds = Subset(train_ds, train_indices)
    
#     # Collate function
#     def collate_fn(batch):
#         input_ids = torch.stack([item['input_ids'] for item in batch])
#         attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
#         result = {
#             'input_ids': input_ids,
#             'attention_mask': attention_mask,
#         }
        
#         # Handle latents
#         if 'latent' in batch[0]:
#             latents = [item['latent'] for item in batch]
#             # Pad if variable length
#             max_len = max(lat.shape[0] for lat in latents)
#             latent_dim = latents[0].shape[-1]
#             padded_latents = torch.zeros(len(latents), max_len, latent_dim)
#             for i, lat in enumerate(latents):
#                 padded_latents[i, :lat.shape[0]] = lat
#             result['latent'] = padded_latents
        
#         return result
    
#     # Create distributed sampler if needed
#     if torch.distributed.is_available() and torch.distributed.is_initialized():
#         train_sampler = DistributedSampler(train_ds, shuffle=True)
#         test_sampler = DistributedSampler(test_ds, shuffle=False)
#     else:
#         train_sampler = None
#         test_sampler = None
    
#     # Create dataloaders
#     train_dl = DataLoader(
#         train_ds,
#         batch_size=config.training.train_batch_size,
#         sampler=train_sampler,
#         shuffle=(train_sampler is None),
#         collate_fn=collate_fn,
#         num_workers=config.data.get('num_workers', 16),
#         pin_memory=True,
#         drop_last=True
#     )
    
#     test_dl = DataLoader(
#         test_ds,
#         batch_size=config.training.eval_batch_size,
#         sampler=test_sampler,
#         shuffle=False,
#         collate_fn=collate_fn,
#         num_workers=config.data.get('num_workers', 16),
#         pin_memory=True,
#         drop_last=False
#     )
    
#     return train_dl, test_dl

















# # # File: latentDLM_mmdit/data_simple.py
# # import json
# # from pathlib import Path
# # import torch
# # import numpy as np
# # from torch.utils.data import Dataset, DataLoader
# # from torch.utils.data.distributed import DistributedSampler
# # import random
# # from datasets import load_dataset
# # import hashlib


# # class SimpleLatentDataset(Dataset):
# #     """Dataset that loads FULL OpenWebText + latents where available."""
    
# #     def __init__(self, config, tokenizer, split="train"):
# #         self.config = config
# #         self.tokenizer = tokenizer
# #         self.split = split
# #         self.max_length = config.model.max_seq_len
        
# #         # Load FULL dataset (all ~8M samples)
# #         print(f"Loading {split} dataset: {config.data.dataset_name}")
        
# #         # Load main text dataset
# #         if config.data.dataset_name == "openwebtext":
# #             if split == "train":
# #                 # Load training split (full dataset)
# #                 self.text_dataset = load_dataset(
# #                     "openwebtext",
# #                     split="train",
# #                     trust_remote_code=True,
# #                     cache_dir=config.data.cache_dir,
# #                     streaming=False
# #                 )
# #                 print(f"Loaded ALL OpenWebText training samples: {len(self.text_dataset)}")
# #             else:
# #                 # Validation split
# #                 test_size = config.data.get('test_size', 5000)
# #                 self.text_dataset = load_dataset(
# #                     "openwebtext",
# #                     split=f"train[-{test_size}:]",
# #                     trust_remote_code=True,
# #                     cache_dir=config.data.cache_dir,
# #                     streaming=False
# #                 )
# #         else:
# #             # Other datasets
# #             self.text_dataset = load_dataset(
# #                 config.data.dataset_name,
# #                 split=split,
# #                 cache_dir=config.data.cache_dir,
# #                 streaming=False
# #             )
        
# #         # Load latent mappings (your 20k processed samples)
# #         self.latent_index = {}
# #         if hasattr(config.data, 'latent_data_root'):
# #             latent_root = Path(config.data.latent_data_root)
            
# #             # Load index file
# #             index_file = latent_root / "train_data.json"
# #             if index_file.exists():
# #                 with open(index_file, 'r', encoding='utf-8') as f:
# #                     latent_data = json.load(f)
                
# #                 # Create better matching: hash full text
# #                 for item in latent_data:
# #                     text = item.get('text', '').strip()
# #                     if text and 'latent_path' in item:
# #                         # Hash the full text for accurate matching
# #                         text_hash = hashlib.md5(text.encode()).hexdigest()
# #                         self.latent_index[text_hash] = {
# #                             'path': latent_root / item['latent_path'],
# #                             'text': text,
# #                             'id': item.get('id', 0)
# #                         }
                
# #                 print(f"Loaded {len(self.latent_index)} latent mappings")
# #             else:
# #                 print(f"Warning: No latent index found at {index_file}")
        
# #         self.has_latents = len(self.latent_index) > 0
# #         self.total_size = len(self.text_dataset)
        
# #         # Track statistics
# #         self.stats = {
# #             'total_samples': self.total_size,
# #             'latent_samples': len(self.latent_index),
# #             'latent_coverage': len(self.latent_index) / self.total_size if self.total_size > 0 else 0
# #         }
        
# #         print(f"Total {split} samples: {self.total_size:,}")
# #         print(f"With latents: {self.stats['latent_samples']:,} ({self.stats['latent_coverage']*100:.2f}%)")
    
# #     def __len__(self):
# #         return self.total_size
    
# #     def __getitem__(self, idx):
# #         # Get text from full dataset
# #         item = self.text_dataset[idx]
# #         text = item.get('text', '').strip()
        
# #         # Tokenize text
# #         tokenized = self.tokenizer(
# #             text,
# #             truncation=True,
# #             padding='max_length',
# #             max_length=self.max_length,
# #             return_tensors='pt'
# #         )
        
# #         result = {
# #             'input_ids': tokenized['input_ids'].squeeze(0),
# #             'attention_mask': tokenized['attention_mask'].squeeze(0),
# #             'has_latent': torch.tensor(0, dtype=torch.bool)  # Default: no latent
# #         }
        
# #         # Try to find matching latent
# #         if self.has_latents and text:
# #             # Hash text for matching
# #             text_hash = hashlib.md5(text.encode()).hexdigest()
            
# #             if text_hash in self.latent_index:
# #                 latent_info = self.latent_index[text_hash]
# #                 latent_path = latent_info['path']
                
# #                 if latent_path.exists():
# #                     try:
# #                         latent = np.load(latent_path)
# #                         latent_tensor = torch.from_numpy(latent).float()
                        
# #                         # Ensure correct shape: [seq_len, latent_dim]
# #                         if latent_tensor.dim() == 1:
# #                             latent_tensor = latent_tensor.unsqueeze(0)  # [1, dim]
                        
# #                         result['latent'] = latent_tensor
# #                         result['has_latent'] = torch.tensor(1, dtype=torch.bool)
                        
# #                     except Exception as e:
# #                         # Silently skip if latent loading fails
# #                         pass
        
# #         return result


# # def collate_latent_fn(batch, config):
# #     """Collate function that handles variable latent presence."""
# #     input_ids = torch.stack([item['input_ids'] for item in batch])
# #     attention_mask = torch.stack([item['attention_mask'] for item in batch])
# #     has_latent = torch.stack([item['has_latent'] for item in batch])
    
# #     result = {
# #         'input_ids': input_ids,
# #         'attention_mask': attention_mask,
# #         'has_latent': has_latent,
# #     }
    
# #     # Check if any sample has latents
# #     any_has_latent = has_latent.any().item()
    
# #     if any_has_latent:
# #         latent_dim = config.model.get('latent_dim', 768)
        
# #         # Collect latents (use zeros for samples without latents)
# #         latents = []
# #         for i, item in enumerate(batch):
# #             if 'latent' in item:
# #                 latents.append(item['latent'])
# #             else:
# #                 # Zero latent for samples without latents
# #                 latents.append(torch.zeros(1, latent_dim))
        
# #         # Pad to same length
# #         max_len = max(lat.shape[0] for lat in latents)
# #         padded_latents = torch.zeros(len(latents), max_len, latent_dim)
        
# #         for i, lat in enumerate(latents):
# #             padded_latents[i, :lat.shape[0]] = lat
        
# #         result['latent'] = padded_latents
# #         result['latent_mask'] = (padded_latents != 0).any(dim=-1)  # Mask for actual latents
    
# #     return result


# # def get_simple_dataloaders(config, tokenizer):
# #     """Main dataloader function - loads FULL dataset."""
    
# #     print("=" * 60)
# #     print("Creating dataloaders with FULL dataset + latent conditioning")
# #     print("=" * 60)
    
# #     # Create datasets
# #     train_ds = SimpleLatentDataset(config, tokenizer, split="train")
    
# #     # Create validation dataset
# #     if hasattr(config.data, 'dataset_name'):
# #         if config.data.dataset_name == "openwebtext":
# #             # Use last portion of training data for validation
# #             test_size = config.data.get('test_size', 5000)
# #             print(f"Using last {test_size} samples for validation")
            
# #             # Create validation subset
# #             from torch.utils.data import Subset
# #             val_indices = list(range(len(train_ds) - test_size, len(train_ds)))
# #             test_ds = Subset(train_ds, val_indices)
            
# #             # Trim training dataset
# #             train_indices = list(range(len(train_ds) - test_size))
# #             train_ds = Subset(train_ds, train_indices)
# #         else:
# #             # Other datasets may have separate validation split
# #             test_ds = SimpleLatentDataset(config, tokenizer, split="validation")
# #     else:
# #         # Fallback: random split
# #         print("Creating random validation split")
# #         total_size = len(train_ds)
# #         val_size = min(5000, total_size // 10)
# #         indices = list(range(total_size))
# #         random.shuffle(indices)
# #         val_indices = indices[:val_size]
# #         train_indices = indices[val_size:]
        
# #         from torch.utils.data import Subset
# #         test_ds = Subset(train_ds, val_indices)
# #         train_ds = Subset(train_ds, train_indices)
    
# #     print(f"Training samples: {len(train_ds):,}")
# #     print(f"Validation samples: {len(test_ds):,}")
    
# #     # Create partial collate function with config
# #     from functools import partial
# #     collate_fn = partial(collate_latent_fn, config=config)
    
# #     # Create distributed sampler if needed
# #     if torch.distributed.is_available() and torch.distributed.is_initialized():
# #         train_sampler = DistributedSampler(train_ds, shuffle=True)
# #         test_sampler = DistributedSampler(test_ds, shuffle=False)
# #     else:
# #         train_sampler = None
# #         test_sampler = None
    
# #     # Create dataloaders
# #     train_dl = DataLoader(
# #         train_ds,
# #         batch_size=config.training.train_batch_size,
# #         sampler=train_sampler,
# #         shuffle=(train_sampler is None),
# #         collate_fn=collate_fn,
# #         num_workers=config.data.get('num_workers', 16),
# #         pin_memory=True,
# #         drop_last=True,
# #         persistent_workers=True
# #     )
    
# #     test_dl = DataLoader(
# #         test_ds,
# #         batch_size=config.training.eval_batch_size,
# #         sampler=test_sampler,
# #         shuffle=False,
# #         collate_fn=collate_fn,
# #         num_workers=config.data.get('num_workers', 2),
# #         pin_memory=True,
# #         drop_last=False
# #     )
    
# #     return train_dl, test_dl





# File: latentDLM_mmdit/data_simple.py (COMPLETE FIXED VERSION)
import json
import pickle
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset
import os

class PreTokenizedLatentDataset(Dataset):
    """Dataset that loads pre-tokenized data with correct latent paths"""
    
    def __init__(self, tokenized_pkl_path, check_latents=True, latent_data_root=None):
        self.tokenized_pkl_path = Path(tokenized_pkl_path)
        
        # CRITICAL FIX: Set the CORRECT base directory for latents
        if latent_data_root is None:
            # Go from: /preprocessed_data_parallel/train_data_tokenized.pkl
            # To: /preprocessed_data/e5_1024d_full/
            self.latent_base_dir = self.tokenized_pkl_path.parent.parent / "preprocessed_data" / "e5_1024d_full"
        else:
            self.latent_base_dir = Path(latent_data_root)
        
        print(f"Pre-tokenized file: {self.tokenized_pkl_path}")
        print(f"Latent base directory: {self.latent_base_dir}")
        print(f"Latent dir exists: {self.latent_base_dir.exists()}")
        
        # Load pre-tokenized data
        print(f"Loading pre-tokenized data...")
        with open(self.tokenized_pkl_path, 'rb') as f:
            self.data = pickle.load(f)
        
        print(f"Loaded {len(self.data):,} pre-tokenized samples")
        
        # Optional: Check latents exist
        if check_latents:
            self._validate_latents()
    
    def _validate_latents(self):
        """Check that latent files exist - CORRECTED PATHS"""
        print(f"Validating latent paths...")
        missing = 0
        found = 0
        
        for i, item in enumerate(self.data[:100]):  # Check first 100 samples
            latent_path = item.get('latent_path', None)
            if latent_path:
                # Try CORRECT path: latent_base_dir + latent_path
                full_path = self.latent_base_dir / latent_path
                
                if full_path.exists():
                    found += 1
                    if found <= 3:  # Print first 3 successes
                        print(f"✓ Found latent: {full_path}")
                else:
                    missing += 1
                    if missing <= 3:  # Print first 3 warnings
                        print(f"✗ Missing latent: {full_path}")
        
        print(f"Latent check: {found} found, {missing} missing (first 100 samples)")
        
        # Also check what the actual latent_path looks like
        if self.data:
            sample_latent = self.data[0].get('latent_path', None)
            if sample_latent:
                print(f"Sample latent_path format: '{sample_latent}'")
                test_path = self.latent_base_dir / sample_latent
                print(f"Full test path: {test_path}")
                print(f"Exists: {test_path.exists()}")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Get pre-tokenized tensors
        input_ids = torch.from_numpy(item['input_ids']).long()
        attention_mask = torch.from_numpy(item['attention_mask']).bool()
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        # Load latent if available - WITH CORRECT PATH
        latent_path = item.get('latent_path', None)
        if latent_path:
            try:
                # CORRECT PATH: Use latent_base_dir
                full_path = self.latent_base_dir / latent_path
                
                if full_path.exists():
                    latent = np.load(full_path, mmap_mode='r')
                    latent_tensor = torch.from_numpy(np.array(latent)).float()
                    
                    if latent_tensor.dim() == 1:
                        latent_tensor = latent_tensor.unsqueeze(0)
                        
                    result['latent'] = latent_tensor
                else:
                    # Create empty latent
                    result['latent'] = torch.zeros(1, 1024)
                    
            except Exception as e:
                print(f"Error loading latent {full_path}: {e}")
                result['latent'] = torch.zeros(1, 1024)
        else:
            # No latent_path in data
            result['latent'] = torch.zeros(1, 1024)
        
        return result


def get_preprocessed_dataloaders(config):
    """Get dataloaders using pre-tokenized data - WITH CORRECT LATENT PATHS"""
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    
    # Get rank info
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    print(f"Rank {rank}: Loading pre-tokenized data...")
    
    # Get preprocessed file path from config
    preprocessed_file = config.data.get("preprocessed_file", 
        "/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data_parallel/train_data_tokenized.pkl")
    
    # Get latent data root from config (where the actual .npy files are)
    latent_data_root = config.data.get("latent_data_root",
        "/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data/e5_1024d_full")
    
    preprocessed_path = Path(preprocessed_file)
    
    if not preprocessed_path.exists():
        raise FileNotFoundError(
            f"Preprocessed file not found: {preprocessed_path}\n"
            "Please run the parallel preprocessing script first."
        )
    
    # Load dataset with CORRECT latent paths
    dataset = PreTokenizedLatentDataset(
        tokenized_pkl_path=preprocessed_path,
        latent_data_root=latent_data_root,
        check_latents=True
    )
    
    print(f"Rank {rank}: Loaded {len(dataset):,} pre-tokenized samples")
    
    # Split into train/val (95/5)
    from torch.utils.data import random_split
    train_size = int(0.95 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_ds, test_ds = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.training.seed)
    )
    
    print(f"Rank {rank}: Split into {len(train_ds)} train, {len(test_ds)} val samples")
    
    # Get num_workers from config (should be low for preprocessed data)
    num_workers = config.data.get("num_workers", 8)
    
    print(f"Rank {rank}: Using {num_workers} dataloader workers")
    
    # Collate function
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if 'latent' in batch[0]:
            latents = []
            latent_dim = config.model.get('latent_dim', 1024)
            
            for item in batch:
                if 'latent' in item:
                    latents.append(item['latent'])
                else:
                    latents.append(torch.zeros(1, latent_dim))
            
            # Find max length and pad
            max_len = max(lat.shape[0] for lat in latents) if latents else 1
            padded_latents = torch.zeros(len(latents), max_len, latent_dim)
            
            for i, lat in enumerate(latents):
                padded_latents[i, :lat.shape[0]] = lat
            
            result['latent'] = padded_latents
        
        return result
    
    # Distributed sampler
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.training.seed
        )
    else:
        train_sampler = None
    
    # DataLoader configuration
    dataloader_kwargs = {
        'batch_size': config.training.train_batch_size,
        'sampler': train_sampler,
        'shuffle': (train_sampler is None),
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True,
    }
    
    # Performance optimizations
    if num_workers > 0:
        dataloader_kwargs.update({
            'prefetch_factor': 2,
            'persistent_workers': True,
        })
    
    train_dl = DataLoader(train_ds, **dataloader_kwargs)
    
    # Validation DataLoader
    test_kwargs = dataloader_kwargs.copy()
    test_kwargs.update({
        'batch_size': config.training.eval_batch_size,
        'sampler': None,
        'shuffle': False,
        'num_workers': min(2, num_workers),
        'drop_last': False,
    })
    
    test_dl = DataLoader(test_ds, **test_kwargs)
    
    print(f"Rank {rank}: DataLoader ready (using pre-tokenized data)")
    
    return train_dl, test_dl


def get_simple_dataloaders(config, tokenizer=None):
    """Get dataloaders - automatically chooses preprocessed or regular"""
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    
    # Get rank info
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    print(f"Rank {rank}: Loading data...")
    
    # Check if we should use preprocessed data
    use_preprocessed = config.data.get("use_preprocessed", False)
    
    if use_preprocessed:
        print(f"Rank {rank}: Using pre-tokenized data")
        train_dl, test_dl = get_preprocessed_dataloaders(config)
    else:
        print(f"Rank {rank}: Using on-the-fly tokenization")
        if tokenizer is None:
            raise ValueError("Tokenizer required for on-the-fly tokenization")
        train_dl, test_dl = get_regular_dataloaders(config, tokenizer)
    
    return train_dl, test_dl


# For backward compatibility - keep the old SimpleLatentDataset
class SimpleLatentDataset(Dataset):
    """Old dataset for on-the-fly tokenization (backward compatibility)"""
    
    def __init__(self, json_path, tokenizer, max_length=512, max_samples=None):
        self.json_path = Path(json_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.base_dir = self.json_path.parent
        
        # Load data
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        if max_samples:
            data = data[:max_samples]
        
        self.texts = [item.get('text', '') for item in data]
        self.latent_paths = [item.get('latent_path', None) for item in data]
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        # Tokenize on-the-fly
        tokenized = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': tokenized['input_ids'].squeeze(0),
            'attention_mask': tokenized['attention_mask'].squeeze(0),
        }
        
        # Load latent
        latent_path = self.latent_paths[idx]
        if latent_path:
            try:
                full_path = self.base_dir / latent_path
                latent = np.load(full_path, mmap_mode='r')
                latent_tensor = torch.from_numpy(np.array(latent)).float()
                if latent_tensor.dim() == 1:
                    latent_tensor = latent_tensor.unsqueeze(0)
                result['latent'] = latent_tensor
            except Exception:
                pass
        
        return result


def get_regular_dataloaders(config, tokenizer):
    """Get dataloaders with on-the-fly tokenization (old method)"""
    from torch.utils.data import DataLoader, DistributedSampler
    import torch.distributed as dist
    
    # Get rank info
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    
    print(f"Rank {rank}: Creating dataloaders with on-the-fly tokenization...")
    
    # Create dataset
    train_ds = SimpleLatentDataset(
        config.data.data_files.train,
        tokenizer,
        max_length=config.model.max_seq_len,
        max_samples=config.data.get('max_samples', None)
    )
    
    print(f"Rank {rank}: Loaded {len(train_ds)} total samples")
    
    # Split into train/val
    from torch.utils.data import Subset
    val_size = min(1000, len(train_ds) // 10)
    val_indices = list(range(val_size))
    test_ds = Subset(train_ds, val_indices)
    
    train_indices = list(range(val_size, len(train_ds)))
    train_ds = Subset(train_ds, train_indices)
    
    print(f"Rank {rank}: Split into {len(train_ds)} train, {len(test_ds)} val samples")
    
    # Use config num_workers
    num_workers = config.data.get('num_workers', 128)
    
    print(f"Rank {rank}: Using {num_workers} dataloader workers")
    
    # Collate function
    def collate_fn(batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        
        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }
        
        if 'latent' in batch[0]:
            latents = []
            latent_dim = config.model.get('latent_dim', 1024)
            
            for item in batch:
                if 'latent' in item:
                    latents.append(item['latent'])
                else:
                    latents.append(torch.zeros(1, latent_dim))
            
            # Find max length
            max_len = max(lat.shape[0] for lat in latents)
            padded_latents = torch.zeros(len(latents), max_len, latent_dim)
            
            for i, lat in enumerate(latents):
                padded_latents[i, :lat.shape[0]] = lat
            
            result['latent'] = padded_latents
        
        return result
    
    # Distributed sampler
    if dist.is_available() and dist.is_initialized():
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=config.training.seed
        )
    else:
        train_sampler = None
    
    # DataLoader configuration
    dataloader_kwargs = {
        'batch_size': config.training.train_batch_size,
        'sampler': train_sampler,
        'shuffle': (train_sampler is None),
        'collate_fn': collate_fn,
        'num_workers': num_workers,
        'pin_memory': True,
        'drop_last': True,
    }
    
    # Performance optimizations
    if num_workers > 0:
        dataloader_kwargs.update({
            'prefetch_factor': 2,
            'persistent_workers': True,
        })
    
    train_dl = DataLoader(train_ds, **dataloader_kwargs)
    
    # Validation DataLoader
    test_kwargs = dataloader_kwargs.copy()
    test_kwargs.update({
        'batch_size': config.training.eval_batch_size,
        'sampler': None,
        'shuffle': False,
        'num_workers': min(8, num_workers // 2),
        'drop_last': False,
    })
    
    test_dl = DataLoader(test_ds, **test_kwargs)
    
    print(f"Rank {rank}: DataLoader ready")
    
    return train_dl, test_dl