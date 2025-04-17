import torch
import gc

def clear_gpu_memory():
    if torch.cuda.is_available():
        # Clear PyTorch cache
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        gc.collect()
        
        # Clear cache again after GC
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Reset peak memory stats
        torch.cuda.reset_peak_memory_stats()
        
        print("GPU memory cleared successfully")
        print(f"Current GPU memory usage: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    else:
        print("No GPU available")

if __name__ == "__main__":
    clear_gpu_memory() 