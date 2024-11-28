from benchmarks.benchmark_runner import BenchmarkRunner

def main():
    runner = BenchmarkRunner()
    
    algorithms = ["matrix_multiplication", "monte_carlo", "mandelbrot"]
    input_sizes = [5, 10, 15] 
    
    for framework in runner.implementations.keys():
        print(f"\nRunning benchmarks for {framework}:")
        for algorithm in algorithms:
            if algorithm in runner.implementations[framework]:
                for input_size in input_sizes:
                    print(f"\nRunning {framework} - {algorithm} with size {input_size}")
                    try:
                        runner.run_benchmark(framework, algorithm, input_size)
                        print(f"✓ Completed successfully")
                    except Exception as e:
                        print(f"✗ Failed: {str(e)}")
    
    runner.save_results("results/benchmark_results.json")
    print("\nBenchmark completed. Results saved to results/benchmark_results.json")

if __name__ == "__main__":
    main()