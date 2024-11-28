#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void monte_carlo_kernel(float* results, int iterations, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < iterations) {
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        
        results[idx] = (x*x + y*y <= 1.0f) ? 1.0f : 0.0f;
    }
}

extern "C" float estimate_pi(int iterations) {
    float *d_results, *h_results;
    h_results = (float*)malloc(iterations * sizeof(float));
    cudaMalloc(&d_results, iterations * sizeof(float));

    int threadsPerBlock = 256;
    int blocks = (iterations + threadsPerBlock - 1) / threadsPerBlock;

    monte_carlo_kernel<<<blocks, threadsPerBlock>>>(d_results, iterations, time(NULL));

    cudaMemcpy(h_results, d_results, iterations * sizeof(float), cudaMemcpyDeviceToHost);

    float sum = 0.0f;
    for (int i = 0; i < iterations; i++) {
        sum += h_results[i];
    }

    float pi = 4.0f * sum / iterations;

    free(h_results);
    cudaFree(d_results);

    return pi;
}