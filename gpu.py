import pycuda.autoinit
import pycuda.driver as drv
import numpy as np
from pycuda.compiler import SourceModule

def gpu_speed_test():
    mod = SourceModule("""
    __global__ void multiply_them(float *dest, float *a, float *b)
    {
      const int i = threadIdx.x + blockDim.x * blockIdx.x;
      dest[i] = a[i] * b[i];
    }
    """)

    multiply_them = mod.get_function("multiply_them")

    # Create large arrays
    size = 10000000
    a = np.random.randn(size).astype(np.float32)
    b = np.random.randn(size).astype(np.float32)
    dest = np.zeros_like(a)

    # Allocate memory on the device
    a_gpu = drv.mem_alloc(a.nbytes)
    b_gpu = drv.mem_alloc(b.nbytes)
    dest_gpu = drv.mem_alloc(dest.nbytes)

    # Copy data to GPU
    drv.memcpy_htod(a_gpu, a)
    drv.memcpy_htod(b_gpu, b)

    # Execute program on device
    start_time = drv.Event()
    end_time = drv.Event()
    start_time.record()
    
    multiply_them(dest_gpu, a_gpu, b_gpu, block=(400,1,1), grid=(size // 400,1))
    
    end_time.record()
    end_time.synchronize()
    
    # Time calculation
    print(f"GPU operation took {start_time.time_till(end_time)} milliseconds")

# Call the function
gpu_speed_test()
