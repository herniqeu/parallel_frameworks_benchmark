import cupy as cp

def mandelbrot(width, height, max_iter):
    """
    CuPy implementation of Mandelbrot set
    Args:
        width: width of the image
        height: height of the image
        max_iter: maximum number of iterations
    Returns:
        numpy array: mandelbrot set values
    """
    y, x = cp.ogrid[-1.4:1.4:height*1j, -2:0.8:width*1j]
    c = x + y*1j
    z = c
    divtime = max_iter + cp.zeros(z.shape, dtype=int)

    for i in range(max_iter):
        z = z**2 + c
        diverge = z*cp.conj(z) > 2**2
        div_now = diverge & (divtime == max_iter)
        divtime[div_now] = i
        z[diverge] = 2

    return cp.asnumpy(divtime)