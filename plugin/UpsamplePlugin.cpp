#include <iostream>
#include "UpsmapleKernel.h"

#define scale_factor  2
#define input_b  3
#define input_c  2
#define input_h  3
#define input_w  3


void plugin_test()
{
    std::cout << "Hello CUDA" << std::endl;
    // showProperties();

    


    // int intput_size = input_b*input_c*input_h*input_w;
    // int output_size = input_b*input_c*input_h*scale_factor*input_w*scale_factor;
    float inputs[54] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0, 
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0,
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0, 
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0,
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0,
                        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 ,9.0};
    float outputs[216];

    int a = UpsampleInference(inputs, outputs, scale_factor,input_b,input_c,input_h,input_w, false);

    printf("end");
}

// #include <pybind11/pybind11.h>
// namespace py = pybind11;

// int add(int i, int j)
// {
//     return i+j;
// }

// PYBIND11_MODULE(demo, m) {
//     m.doc() = "pybind11 plugin demo"; // optional module docstring
//     m.def("add", &add, "A function which adds two numbers");
// }