import time
import numpy as np
import psutil
import json
from datetime import datetime
import GPUtil

class BenchmarkRunner:
    def __init__(self):
        self.results = []
        self.system_info = self._collect_system_info()
        
        # Initialize implementations dict
        self.implementations = {}
        
        # Try loading each implementation type
        openmp_impls = self._load_openmp_implementations()
        if openmp_impls:
            self.implementations['openmp'] = openmp_impls
        
        cuda_impls = self._load_cuda_implementations()
        if cuda_impls:
            self.implementations['cuda'] = cuda_impls
        
        cupy_impls = self._load_cupy_implementations()
        if cupy_impls:
            self.implementations['cupy'] = cupy_impls
        
        if not self.implementations:
            print("Warning: No implementations were successfully loaded!")
    
    def _collect_system_info(self):
        gpus = GPUtil.getGPUs()
        gpu_info = gpus[0].name if gpus else "N/A"
        
        # Get CPU info from /proc/cpuinfo on Linux
        cpu_info = "Unknown"
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        cpu_info = line.split(':')[1].strip()
                        break
        except:
            pass
        
        uname = psutil.os.uname()
        
        return {
            "cpu": cpu_info,
            "gpu": gpu_info,
            "memory": f"{psutil.virtual_memory().total / (1024**3):.2f}GB",
            "os": f"{uname.sysname} {uname.release}",
            "cpu_cores": psutil.cpu_count(logical=False),
            "cpu_threads": psutil.cpu_count(logical=True)
        }
    
    def _load_openmp_implementations(self):
        try:
            from implementations.openmp.matrix_multiplication import matrix_multiplication
            from implementations.openmp.monte_carlo import monte_carlo_pi as monte_carlo
            from implementations.openmp.mandelbrot import mandelbrot_set as mandelbrot
            return {
                'matrix_multiplication': matrix_multiplication,
                'monte_carlo': monte_carlo,
                'mandelbrot': mandelbrot
            }
        except ImportError as e:
            print(f"Warning: OpenMP implementations not available: {e}")
            return {}

    def _load_cuda_implementations(self):
        try:
            from implementations.cuda.matrix_multiplication import matrix_multiplication
            from implementations.cuda.monte_carlo import monte_carlo
            from implementations.cuda.mandelbrot import mandelbrot
            return {
                'matrix_multiplication': matrix_multiplication,
                'monte_carlo': monte_carlo,
                'mandelbrot': mandelbrot
            }
        except ImportError as e:
            print(f"Warning: CUDA implementations not available: {e}")
            return {}

    def _load_cupy_implementations(self):
        try:
            from implementations.cupy.matrix_multiplication import matrix_multiplication
            from implementations.cupy.monte_carlo import monte_carlo
            from implementations.cupy.mandelbrot import mandelbrot
            return {
                'matrix_multiplication': matrix_multiplication,
                'monte_carlo': monte_carlo,
                'mandelbrot': mandelbrot
            }
        except ImportError as e:
            print(f"Warning: CuPy implementations not available: {e}")
            return {}

    def run_benchmark(self, framework, algorithm, input_size, num_runs=5):
        if framework not in self.implementations:
            print(f"Warning: Framework {framework} not available")
            return
        if algorithm not in self.implementations[framework]:
            print(f"Warning: Algorithm {algorithm} not available for {framework}")
            return

        implementation = self.implementations[framework][algorithm]
        
        results = []
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs}...")
            
            # Measure execution time and memory usage
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Run the implementation with appropriate arguments
            if algorithm == 'matrix_multiplication':
                input_matrix = np.random.rand(input_size, input_size).astype(np.float32)
                output_matrix = np.zeros((input_size, input_size), dtype=np.float32)
                result = implementation(input_matrix, output_matrix, input_size)
            elif algorithm == 'monte_carlo':
                result = implementation(input_size)
            else:  # mandelbrot
                output = np.zeros((input_size, input_size), dtype=np.uint8)
                result = implementation(output, input_size, input_size, 1000)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / (1024 * 1024)  # MB
            
            # Collect metrics
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            memory_usage = memory_after - memory_before
            
            results.append({
                "execution_time_ms": execution_time,
                "memory_usage_mb": memory_usage,
                "timestamp": datetime.now().isoformat(),
                "hardware_metrics": {}
            })
        
        # Store benchmark results
        self.results.append({
            "framework": framework,
            "algorithm": algorithm,
            "input_size": input_size,
            "iterations": results
        })
    
    def _prepare_input(self, algorithm, size):
        if algorithm == 'matrix_multiplication':
            return np.random.rand(size, size).astype(np.float32)
        elif algorithm == 'monte_carlo':
            return size  # Just return the number of points
        else:  # mandelbrot
            return size  # Return the grid size
    
    def save_results(self, filename):
        with open(filename, 'w') as f:
            json.dump({
                "system_info": self.system_info,
                "benchmarks": self.results
            }, f, indent=2)