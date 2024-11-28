#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

static PyObject* mandelbrot_set(PyObject* self, PyObject* args) {
    PyArrayObject *output_array;
    int width, height, max_iter;
    
    if (!PyArg_ParseTuple(args, "O!iii", &PyArray_Type, &output_array, 
                         &width, &height, &max_iter)) {
        return NULL;
    }
    
    if (output_array == NULL) {
        PyErr_SetString(PyExc_ValueError, "Output array is NULL");
        return NULL;
    }
    
    unsigned char* output = (unsigned char*)PyArray_DATA(output_array);
    if (output == NULL) {
        PyErr_SetString(PyExc_ValueError, "Output data buffer is NULL");
        return NULL;
    }
    
    #pragma omp parallel for collapse(2) schedule(dynamic)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double x0 = (x - width/2.0) * 4.0/width;
            double y0 = (y - height/2.0) * 4.0/height;
            
            double x1 = 0;
            double y1 = 0;
            int iteration = 0;
            
            while (x1*x1 + y1*y1 <= 4 && iteration < max_iter) {
                double xtemp = x1*x1 - y1*y1 + x0;
                y1 = 2*x1*y1 + y0;
                x1 = xtemp;
                iteration++;
            }
            
            if (y * width + x < width * height) {
                output[y * width + x] = iteration < max_iter ? 
                    (unsigned char)(255 * iteration / max_iter) : 0;
            }
        }
    }
    
    Py_RETURN_NONE;
}

static PyMethodDef MandelbrotMethods[] = {
    {"mandelbrot_set", mandelbrot_set, METH_VARARGS, 
     "Mandelbrot set calculation using OpenMP"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef mandelbrotmodule = {
    PyModuleDef_HEAD_INIT,
    "mandelbrot",
    NULL,
    -1,
    MandelbrotMethods
};

PyMODINIT_FUNC PyInit_mandelbrot(void) {
    import_array();  // Initialize NumPy
    return PyModule_Create(&mandelbrotmodule);
}