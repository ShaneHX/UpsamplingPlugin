import ctypes
import os

import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt
import numpy as np

TRT_LOGGER = trt.Logger()


CLIP_PLUGIN_LIBRARY = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'build/libupsampleplugin.so'
)
# ctypes.CDLL(CLIP_PLUGIN_LIBRARY)

# CDLL("/usr/lib/x86_64-linux-gnu/libgomp.so.1", mode=RTLD_GLOBAL)
lib = ctypes.cdll.LoadLibrary(CLIP_PLUGIN_LIBRARY)

trt.init_libnvinfer_plugins(TRT_LOGGER, '')
PLUGIN_CREATORS = trt.get_plugin_registry().plugin_creator_list
# for s_creator in PLUGIN_CREATORS:
#     print(s_creator.name)

def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    print("binding shape: {}, {}",engine.get_binding_shape(0),engine.get_binding_shape(1))
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream


def get_trt_plugin(plugin_name):
    plugin = None
    for plugin_creator in PLUGIN_CREATORS:
        if plugin_creator.name == plugin_name:
            scale_factor_field = trt.PluginField("scaleFactor", np.array([2], dtype=np.int8), trt.PluginFieldType.INT8)
            align_corners_field = trt.PluginField("alignCorners", np.array([0], dtype=np.int8), trt.PluginFieldType.INT8)
            field_collection = trt.PluginFieldCollection([align_corners_field, scale_factor_field])
            plugin = plugin_creator.create_plugin(name=plugin_name, field_collection=field_collection)
    return plugin




def build_engine():
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network, trt.CaffeParser() as parser:
        builder.max_batch_size = 1
        builder.max_workspace_size = 2**20
        input_layer = network.add_input(name="input_layer", dtype=trt.float32, shape=(1, 13, 3, 3))
        # bn_w = []
        # bn = network.add_scale(input=[input_layer], mode=trt.ScaleMode.CHANNEL, )
        upsample = network.add_plugin_v2(inputs=[input_layer], plugin=get_trt_plugin("UpsamplePlugin"))
        upsample.get_output(0).name = "outputs"
        network.mark_output(upsample.get_output(0))

        return builder.build_cuda_engine(network)


def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()


def main():

    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, \
                    1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0
                    ])
    print(arr)
    with build_engine() as engine:
        # Build an engine, allocate buffers and create a stream.
        # For more information on buffer allocation, refer to the introductory samples.
        h_input, d_input, h_output, d_output, stream = allocate_buffers(engine)
        np.copyto(h_input, arr)
        # print("debug")
        with engine.create_execution_context() as context:
            # case_num = load_normalized_test_case(data_path, inputs[0].host, mean)
            # For more information on performing inference, refer to the introductory samples.
            # The common.do_inference function will return a list of outputs - we only have one in this case.
            do_inference(context, h_input, d_input, h_output, d_output, stream)
            print(h_output)



if __name__ == "__main__":
    main()