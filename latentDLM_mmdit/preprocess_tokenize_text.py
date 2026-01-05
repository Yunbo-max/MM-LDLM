# File: preprocess_tokenize_text_parallel.py
import json
import pickle
from pathlib import Path
import os
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time

# ========== CRITICAL: Set this BEFORE any imports ==========
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def tokenize_worker(args):
    """Worker function for parallel tokenization"""
    chunk, tokenizer_path, max_length = args
    
    # Import tokenizer INSIDE worker (critical!)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    results = []
    for item in chunk:
        text = item.get('text', '')
        tokenized = tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        results.append({
            'input_ids': tokenized['input_ids'].numpy().astype(np.int32)[0],
            'attention_mask': tokenized['attention_mask'].numpy().astype(np.bool_)[0],
            'latent_path': item.get('latent_path', None),
        })
    
    return results

def pre_tokenize_dataset_parallel(json_path, tokenizer_name, max_length=512, 
                                 output_dir=None, num_workers=55, chunk_size=200):
    """PARALLEL pre-tokenization using all CPU cores."""
    
    json_path = Path(json_path)
    if output_dir is None:
        output_dir = json_path.parent / "preprocessed_parallel"
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    print("=== PARALLEL PREPROCESSING ===")
    print(f"Workers: {num_workers}")
    print(f"Chunk size: {chunk_size}")
    
    # Load data
    print(f"Loading data from {json_path}...")
    start_time = time.time()
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print(f"✓ Loaded {len(data):,} samples in {time.time()-start_time:.1f}s")
    
    # Create chunks for parallel processing
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunks.append(data[i:i+chunk_size])
    
    print(f"Created {len(chunks)} chunks")
    
    # Prepare arguments for workers
    worker_args = [(chunk, tokenizer_name, max_length) for chunk in chunks]
    
    # PARALLEL PROCESSING
    print(f"\nStarting parallel tokenization with {num_workers} workers...")
    print("You should see ~5500% CPU usage in 'top'")
    
    all_results = []
    processed = 0
    
    # Use multiprocessing Pool
    with mp.Pool(
        processes=num_workers,
        maxtasksperchild=20  # Refresh workers to avoid memory leaks
    ) as pool:
        try:
            # Process chunks in parallel
            with tqdm(total=len(chunks), desc="Processing") as pbar:
                for result in pool.imap_unordered(tokenize_worker, worker_args, chunksize=1):
                    all_results.extend(result)
                    processed += len(result)
                    pbar.update(1)
                    
                    # Update speed
                    elapsed = time.time() - start_time
                    if elapsed > 0:
                        pbar.set_postfix({
                            "samples": f"{processed:,}",
                            "speed": f"{processed/elapsed:.0f}/s"
                        })
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted! Saving partial results...")
            pool.terminate()
            pool.join()
        except Exception as e:
            print(f"\n❌ Error: {e}")
            pool.terminate()
            pool.join()
            raise
    
    # Save results
    output_file = output_dir / f"{json_path.stem}_tokenized.pkl"
    print(f"\nSaving {len(all_results):,} tokenized samples to {output_file}...")
    
    with open(output_file, 'wb') as f:
        pickle.dump(all_results, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    # Save tokenizer info
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer_info = {
        'vocab_size': tokenizer.vocab_size,
        'max_length': max_length,
        'pad_token_id': tokenizer.pad_token_id,
        'cls_token_id': tokenizer.cls_token_id,
        'sep_token_id': tokenizer.sep_token_id,
        'mask_token_id': getattr(tokenizer, 'mask_token_id', None),
    }
    
    with open(output_dir / "tokenizer_info.json", 'w') as f:
        json.dump(tokenizer_info, f, indent=2)
    
    # Performance stats
    total_time = time.time() - start_time
    print(f"\n✅ DONE!")
    print(f"   Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   Samples: {len(all_results):,}")
    print(f"   Speed: {len(all_results)/total_time:.0f} samples/sec")
    print(f"   CPU cores used: {num_workers}")
    print(f"   Output: {output_file}")
    
    return output_file, tokenizer_info

if __name__ == "__main__":
    # THIS IS CRITICAL FOR MULTIPROCESSING
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True, 
                       help="Path to input JSON file")
    parser.add_argument("--tokenizer", type=str, 
                       default="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/data/huggingface/tokenizers/bert-base-uncased",
                       help="Tokenizer path")
    parser.add_argument("--max_length", type=int, default=512, 
                       help="Maximum sequence length")
    parser.add_argument("--output_dir", type=str, 
                       default="/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data",
                       help="Output directory")
    parser.add_argument("--workers", type=int, default=55,
                       help="Number of worker processes (use 53 to leave 2 cores for system)")
    parser.add_argument("--chunk_size", type=int, default=200,
                       help="Samples per chunk (default: 200)")
    
    args = parser.parse_args()
    
    pre_tokenize_dataset_parallel(
        json_path=args.json_path,
        tokenizer_name=args.tokenizer,
        max_length=args.max_length,
        output_dir=args.output_dir,
        num_workers=args.workers,
        chunk_size=args.chunk_size
    )