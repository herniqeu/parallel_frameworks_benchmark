#include <cuda_runtime.h>

__device__ int mandelbrot(float x0, float y0, int max_iter) {
    float x = 0.0f, y = 0.0f;
    int iter = 0;
    
    while (x*x + y*y <= 4.0f && iter < max_iter) {
        float xtemp = x*x - y*y + x0;
        y = 2*x*y + y0;
        x = xtemp;
        iter++;
    }
    return iter;
}

__global__ void mandelbrot_kernel(int* output, int width, int height, 
                                float x_min, float x_max, float y_min, float y_max, 
                                int max_iter) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < height && col < width) {
        float x0 = x_min + (x_max - x_min) * col / (width - 1);
        float y0 = y_min + (y_max - y_min) * row / (height - 1);
        
        output[row * width + col] = mandelbrot(x0, y0, max_iter);
    }
}