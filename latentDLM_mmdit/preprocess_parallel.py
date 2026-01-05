# File: latentDLM_mmdit/monitor_preprocessing.py
import psutil
import time
import threading
import json
from pathlib import Path

def monitor_resources(interval=5):
    """Monitor CPU, memory, and disk usage"""
    print("Monitoring system resources...")
    print("Time | CPU% | Mem% | Disk% | Samples/s")
    print("-" * 50)
    
    start_time = time.time()
    last_update = start_time
    last_samples = 0
    
    try:
        while True:
            # Check output file for progress
            output_dir = Path("/inspire/hdd/global_user/zhangjiaquan-253108540222/latent/MM-LDLM/preprocessed_data_fast")
            metadata_file = output_dir / "progress.json"
            
            if metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    progress = json.load(f)
                current_samples = progress.get('processed', 0)
            else:
                current_samples = 0
            
            # Calculate samples per second
            current_time = time.time()
            time_diff = current_time - last_update
            sample_diff = current_samples - last_samples
            
            if time_diff > 0:
                samples_per_sec = sample_diff / time_diff
            else:
                samples_per_sec = 0
            
            # Get system stats
            cpu_percent = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            print(f"{time.time()-start_time:6.1f}s | {cpu_percent:4.1f}% | {mem.percent:4.1f}% | {disk.percent:4.1f}% | {samples_per_sec:7.0f}/s")
            
            last_update = current_time
            last_samples = current_samples
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    # Run monitoring in background thread
    monitor_thread = threading.Thread(target=monitor_resources)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting monitor...")