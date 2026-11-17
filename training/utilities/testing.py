import torch.multiprocessing as mp # type: ignore
import os

def get_safe_num_workers(requested_workers: int = 4) -> int:
    """
    Validates if multiprocessing is supported and can safely initialize.
    Returns 0 if multiprocessing is unavailable, otherwise returns the requested workers.
    """
    # If the user explicitly requested 0, bypass checks
    if requested_workers <= 0:
        return 0

    try:
        # 1. Attempt to set/verify the 'spawn' start method (required for CUDA safety)
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
        
        # 2. Test-spawn a lightweight process to ensure the OS runtime supports it
        test_queue = mp.Queue()
        test_process = mp.Process(target=lambda q: q.put("success"), args=(test_queue,))
        test_process.start()
        test_process.join(timeout=2.0)  # Wait briefly for process execution
        
        if test_process.is_alive():
            # Process hung, terminate it and fall back
            test_process.terminate()
            raise RuntimeError("Multiprocessing test timed out.")
            
        if test_queue.get_nowait() != "success":
            raise RuntimeError("Multiprocessing test failed to return status.")
            
        # Multiprocessing is fully active and functional!
        return requested_workers

    except Exception as e:
        # Fail-safe triggered: Print warning and gracefully run in the main thread
        print(f"[WARNING] Multiprocessing test failed: '{e}'. Falling back to num_workers=0 (single-thread).")
        return 0