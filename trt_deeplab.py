import sys
import os
import copy
import cv2
import time
import torch.nn.functional as F
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from PIL import Image
import torch
import glob

TRT_LOGGER = trt.Logger()

import tensorrt as trt
CLASSES = ('_background', 'LK')

model_input_w = 816
model_input_h = 612

color_list = [[0, 0, 0], [255, 0, 0]]

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine, context):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for i, binding in enumerate(engine):
        size = trt.volume(context.get_binding_shape(i))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, batch_size):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]



class Deeplabv3_engine(object):
    def __init__(self):
        self.engine_path = "./DeepLabV3_F16.trt"
        # self.input_size = [3, 512, 512]
        # self.input_shape = [512, 512]
        # self.image_size = self.input_size[1:]
        self.engine = self.get_engine()
        self.context = self.engine.create_execution_context()
        self.colors = [(0, 0, 0), (128, 0, 0),]

    def img_preprocess(self,src_img):
        img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (model_input_w, model_input_h))
        img = img.astype(np.float32)

        # Normalize
        mean = np.array([123.575, 116.28, 103.53], dtype=np.float32)
        std = np.array([58.395, 57.12, 57.375], dtype=np.float32)

        img = (img - mean) / std
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img, axis=0)  # (1, 3, 612, 816)
        img = np.ascontiguousarray(img, dtype=np.float32)

        return img

    def get_engine(self):
        # If a serialized engine exists, use it instead of building an engine.
        f = open(self.engine_path, 'rb')
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

    def detect(self, image_src, imgpath, cuda_ctx=pycuda.autoinit.context):

        cuda_ctx.push()

        img_in = self.img_preprocess(image_src)

        # 动态输入
        # self.context.active_optimization_profile = 0   #与TensorRT的优化配置文件（Optimization Profiles）相关。
        # 优化配置文件允许你为TensorRT的执行引擎指定多个不同的输入尺寸和数据类型，这样TensorRT就可以为每个配置优化其内部表示，并在运行时基于提供的输入尺寸选择最佳的优化。
        origin_inputshape = self.context.get_binding_shape(0)
        origin_inputshape[0], origin_inputshape[1], origin_inputshape[2], origin_inputshape[3] = img_in.shape

        print("origin_inputshape",origin_inputshape)

        self.context.set_binding_shape(0, (origin_inputshape))  # 若每个输入的size不一样，可根据inputs的size更改对应的context中的size

        inputs, outputs, bindings, stream = allocate_buffers(self.engine, self.context)
        # Do inference
        inputs[0].host = img_in

        start = time.time()
        trt_outputs = do_inference(self.context, bindings=bindings, inputs=inputs, outputs=outputs,
                                          stream=stream, batch_size=1)
        end = time.time()
        print("inference time = ", end - start)
        if cuda_ctx:
            cuda_ctx.pop()

        # nInput = np.sum([self.engine.binding_is_input(i) for i in range(self.engine.num_bindings)])
        # nOutput = self.engine.num_bindings - nInput
        # print("nInput,nOutput",nInput,nOutput)
        # print("trt_putputs",trt_outputs[0].shape)
        #
        # shape = self.context.get_binding_shape(nInput )
        # name = self.engine.get_binding_name(nInput)
        #
        # print("shape,name", shape, name)

        pr = trt_outputs[0].reshape(-1, 612, 816)  ##  c H W
        print("trt_pr", pr.shape)
        pr = pr.transpose(1, 2, 0)*255
        print("trt_pr", type(pr))
        return pr

if __name__ == '__main__':
    imgpath = "/home/myue/002_study/tools/NCNN/test.jpg"
    # img = cv2.imread("img/street.jpg")
    detect_engine = Deeplabv3_engine()

    img = cv2.imread(f"{imgpath}")
    output = detect_engine.detect(img, imgpath)
    # img_name = imgpath.split("/", 1)[1]
    # cv2.imwrite(f"pole_res_trt/{i}_{img_name}", np.float32(output))
    cv2.imwrite("./trt_deeplab_res.jpg",output)