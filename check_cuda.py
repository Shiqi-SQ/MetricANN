import torch

print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("CUDNN enabled:", torch.backends.cudnn.enabled)
print("GPU count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("Current device index:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(0))
