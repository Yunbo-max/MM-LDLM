# File: latentDLM_mmdit/modeling_mmdit.py
from transformers import AutoTokenizer
import torch.nn as nn
from .models.multimodal_mmdit import MultimodalMMDiT
import os


def get_tokenizer(config):
    from transformers import AutoTokenizer
    import os
    
    tokenizer_path = "/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/data/huggingface/tokenizers/bert-base-uncased"
    
    print(f"Checking tokenizer directory: {tokenizer_path}")
    
    # List all files
    for root, dirs, files in os.walk(tokenizer_path):
        for file in files:
            print(f"  {os.path.join(root, file)}")
    
    # Check for required files
    required_files = ['tokenizer.json', 'tokenizer_config.json', 'vocab.txt', 'vocab.json']
    missing = []
    for req in required_files:
        if not os.path.exists(os.path.join(tokenizer_path, req)):
            missing.append(req)
    
    if missing:
        print(f"Warning: Missing tokenizer files: {missing}")
    
    # Try to load anyway
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print("✓ Tokenizer loaded from directory")
    except Exception as e:
        print(f"Failed to load from directory: {e}")
        
        # Try loading as a MODEL directory (it might be a model, not tokenizer)
        print("Trying to load tokenizer from model directory...")
        try:
            # Sometimes tokenizer files are in a subdirectory
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        except Exception as e2:
            print(f"Also failed: {e2}")
            
            # Last resort: Create a simple tokenizer
            print("Creating a fallback tokenizer...")
            from transformers import BertTokenizerFast
            tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    return tokenizer


# def get_tokenizer(config):
#     tokenizer_name = config.data.tokenizer_name
    
#     # Check if tokenizer_name is a local path that exists
#     import os
#     local_path = None
#     if os.path.exists(tokenizer_name):
#         local_path = tokenizer_name
#     else:
#         # Check in local_models directory
#         local_models_dir = os.path.join(os.path.dirname(__file__), "..", "local_models")
#         potential_local_path = os.path.join(local_models_dir, tokenizer_name)
#         if os.path.exists(potential_local_path):
#             local_path = potential_local_path
    
#     if local_path is not None:
#         print(f"Loading tokenizer from local path: {local_path}")
#         tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
#     else:
#         print(f"Loading tokenizer from Hugging Face Hub: {tokenizer_name}")
#         try:
#             tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#         except Exception as e:
#             # Try with offline mode and local files only as fallback
#             print(f"Failed to download tokenizer from HF Hub: {e}")
#             print("Trying to load from cache or local files only...")
#             tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, local_files_only=True)
    
#     if tokenizer.pad_token_id is None:
#         tokenizer.add_special_tokens({"pad_token": "[PAD]"})
#     if tokenizer.mask_token_id is None:
#         tokenizer.add_special_tokens({"mask_token": "[MASK]"})
#     tokenizer.model_max_length = config.model.max_seq_len
#     return tokenizer


def get_model(config, tokenizer, device=None, dtype=None):
    vocab_size = len(tokenizer)
    
    if config.model.type == "multimodal_mmdit":
        print(f"Using Multimodal MMDiT for joint text-latent generation")
        model = MultimodalMMDiT(
            config=config.model,
            vocab_size=vocab_size,
            latent_dim=config.model.get("latent_dim", 768),
            cluster_size=config.model.get("cluster_size", 0)
        )
    else:
        raise ValueError(f"Unknown model type: {config.model.type}. Use 'multimodal_mmdit' for MMDiT training.")

    if device is not None:
        model = model.to(device, dtype=dtype)
    elif dtype is not None:
        model = model.to(dtype=dtype)

    return model