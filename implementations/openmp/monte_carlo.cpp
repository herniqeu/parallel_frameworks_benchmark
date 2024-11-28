#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>
#include <random>

static PyObject* monte_carlo_pi(PyObject* self, PyObject* args) {
    int num_points;
    
    if (!PyArg_ParseTuple(args, "i", &num_points)) {
        return NULL;
    }
    
    int points_inside = 0;
    
    #pragma omp parallel reduction(+:points_inside)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);
        
        #pragma omp for
        for (int i = 0; i < num_points; i++) {
            double x = dis(gen);
            double y = dis(gen);
            if (x*x + y*y <= 1) {
                points_inside++;
            }
        }
    }
    
    double pi = 4.0 * points_inside / num_points;
    return PyFloat_FromDouble(pi);
}

static PyMethodDef MonteCarloPiMethods[] = {
    {"monte_carlo_pi", monte_carlo_pi, METH_VARARGS, "Monte Carlo Pi calculation using OpenMP"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef montecarlomodule = {
    PyModuleDef_HEAD_INIT,
    "monte_carlo",
    NULL,
    -1,
    MonteCarloPiMethods
};

PyMODINIT_FUNC PyInit_monte_carlo(void) {
    return PyModule_Create(&montecarlomodule);
}