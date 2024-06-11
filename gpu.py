import torch
import time

def torch_gpu_speed_test():
    # Check if CUDA is available
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting...")
        return

    # Set device to GPU
    device = torch.device("cuda")

    # Define the size of the tensors
    size = 5000  # You can adjust this size based on your GPU's capability

    # Initialize two random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Warm up GPU
    for _ in range(10):
        _ = torch.mm(a, b)

    torch.cuda.synchronize()  # wait for mm to finish

    start_time = time.time()

    # Perform matrix multiplication
    for _ in range(50):
        c = torch.mm(a, b)

    torch.cuda.synchronize()  # wait for mm to finish

    end_time = time.time()
    duration = end_time - start_time

    print(f"50 matrix multiplications of size {size}x{size} took {duration:.4f} seconds.")

# Run the benchmark
torch_gpu_speed_test()

