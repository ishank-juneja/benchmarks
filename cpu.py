import numpy as np
import time

def cpu_speed_test():
    # Create large arrays
    size = 10000000
    a = np.random.rand(size)
    b = np.random.rand(size)
    
    # Measure time for arithmetic operation
    start_time = time.time()
    c = a * b
    duration = time.time() - start_time
    print(f"Array multiplication (size {size}): {duration:.6f} seconds")
    
    # Sorting
    start_time = time.time()
    sorted_a = np.sort(a)
    duration = time.time() - start_time
    print(f"Array sorting (size {size}): {duration:.6f} seconds")

    # Use numpy to perform a more complex operation
    start_time = time.time()
    d = np.cos(a) + np.sin(b)
    duration = time.time() - start_time
    print(f"Trigonometric operations: {duration:.6f} seconds")

# Call the function
cpu_speed_test()
