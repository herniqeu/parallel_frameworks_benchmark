import cupy as cp

def matrix_multiplication(matrix):
    """
    CuPy implementation of matrix multiplication
    Args:
        matrix: numpy array of shape (N, N)
    Returns:
        result: numpy array of shape (N, N)
    """
    # Convert input to CuPy array
    matrix_gpu = cp.asarray(matrix)
    # Perform multiplication on GPU
    result_gpu = cp.matmul(matrix_gpu, matrix_gpu)
    # Convert back to numpy array
    return cp.asnumpy(result_gpu)