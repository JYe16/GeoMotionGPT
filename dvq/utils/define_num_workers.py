import os
import torch

def define_num_workers(preferred=8):
    """
    Determine the number of workers for DataLoader.
    Returns 0 if CUDA is not available (often implies debugging or local CPU run),
    otherwise returns min(usable_cpu_count, preferred).
    """
    if not torch.cuda.is_available():
        return 0
    
    # Try to get the number of CPUs available to the process (affinity)
    # This is important on clusters (SLURM, etc.) where os.cpu_count() returns total node CPUs
    if hasattr(os, 'sched_getaffinity'):
        try:
            cpu_count = len(os.sched_getaffinity(0))
        except Exception:
            cpu_count = os.cpu_count()
    else:
        cpu_count = os.cpu_count()

    if cpu_count is None:
        return 0
        
    return min(cpu_count, preferred)
