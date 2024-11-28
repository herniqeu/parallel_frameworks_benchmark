import cupy as cp

def monte_carlo(num_points):
    """
    CuPy implementation of Monte Carlo Pi calculation
    Args:
        num_points: number of points to generate
    Returns:
        float: approximation of Pi
    """
    # Generate random points on GPU
    x = cp.random.uniform(0, 1, num_points)
    y = cp.random.uniform(0, 1, num_points)
    
    # Calculate points inside circle
    inside_circle = cp.sum((x * x + y * y) <= 1.0)
    
    # Calculate Pi approximation
    pi_approximation = 4.0 * float(inside_circle) / num_points
    return pi_approximation