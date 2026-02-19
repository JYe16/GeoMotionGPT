import torch


def _is_cuda_device_usable(device_index: int) -> bool:
    try:
        device = torch.device(f'cuda:{device_index}')
        x = torch.randn(1, 4, 8, device=device)
        w = torch.randn(4, 4, 3, device=device)
        _ = torch.nn.functional.conv1d(x, w, padding=1)
        torch.cuda.synchronize(device)
        return True
    except Exception:
        return False

def define_device():
    if torch.cuda.is_available():
        gpu_num = torch.cuda.device_count()
        preferred = gpu_num - 1
        if _is_cuda_device_usable(preferred):
            return torch.device(f'cuda:{preferred}')

        for device_index in range(gpu_num - 1, -1, -1):
            if device_index == preferred:
                continue
            if _is_cuda_device_usable(device_index):
                return torch.device(f'cuda:{device_index}')

        return torch.device('cpu')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')