import torch
import gc
import logging
from typing import Optional

class MemoryManager:
    """Utility class for managing model memory and resources."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        
    def clear_gpu_memory(self):
        """Clear CUDA memory cache."""
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                gc.collect()
                self.logger.debug("GPU memory cache cleared")
            except Exception as e:
                self.logger.warning(f"Failed to clear GPU memory: {e}")
                
    def unload_model(self, model):
        """Safely unload a model from memory."""
        try:
            if hasattr(model, 'cpu'):
                model.cpu()
            del model
            gc.collect()
            self.clear_gpu_memory()
            self.logger.debug("Model unloaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to unload model: {e}")
            
    def get_gpu_memory_info(self) -> dict:
        """Get current GPU memory usage information."""
        if not torch.cuda.is_available():
            return {"error": "CUDA not available"}
            
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = torch.cuda.memory_allocated()
            cached_memory = torch.cuda.memory_reserved()
            
            return {
                "total": total_memory,
                "allocated": allocated_memory,
                "cached": cached_memory,
                "free": total_memory - allocated_memory
            }
        except Exception as e:
            return {"error": str(e)}
            
    def check_gpu_memory_available(self, required_mb: int) -> bool:
        """Check if enough GPU memory is available."""
        if not torch.cuda.is_available():
            return False
            
        try:
            memory_info = self.get_gpu_memory_info()
            if "error" in memory_info:
                return False
                
            free_memory_mb = memory_info["free"] / (1024 * 1024)
            return free_memory_mb >= required_mb
        except Exception as e:
            self.logger.warning(f"Failed to check GPU memory: {e}")
            return False