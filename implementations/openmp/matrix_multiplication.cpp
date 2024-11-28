#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

static PyObject* matrix_multiplication(PyObject* self, PyObject* args) {
    PyArrayObject *input_array, *output_array;
    int N;
    
    if (!PyArg_ParseTuple(args, "O!O!i", &PyArray_Type, &input_array, 
                         &PyArray_Type, &output_array, &N)) {
        return NULL;
    }
    
    float* input = (float*)PyArray_DATA(input_array);
    float* output = (float*)PyArray_DATA(output_array);
    
    #pragma omp parallel for collapse(2)
    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            float sum = 0.0f;
            for(int k = 0; k < N; k++) {
                sum += input[i * N + k] * input[k * N + j];
            }
            output[i * N + j] = sum;
        }
    }
    
    Py_RETURN_NONE;
}

static PyMethodDef MatrixMethods[] = {
    {"matrix_multiplication", matrix_multiplication, METH_VARARGS, 
     "Matrix multiplication using OpenMP"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef matrixmodule = {
    PyModuleDef_HEAD_INIT,
    "matrix_multiplication",
    NULL,
    -1,
    MatrixMethods
};

PyMODINIT_FUNC PyInit_matrix_multiplication(void) {
    import_array();
    return PyModule_Create(&matrixmodule);
}