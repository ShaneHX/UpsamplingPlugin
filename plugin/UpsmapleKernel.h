#ifndef UPSAMPLE_KERNEL_H
#define UPSAMPLE_KERNEL_H

#include <iostream>
#include "NvInfer.h"

int UpsampleInference(
    cudaStream_t stream,
    int n,
    int input_b,
    int input_c,
    int input_h,
    int input_w,
    int scale_factor,
    bool align_corners,
    const void* inputs,
    void* outputs);


#endif
