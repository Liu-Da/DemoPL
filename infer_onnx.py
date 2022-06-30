# -*-coding: utf-8 -*-
import time

import numpy as np
import onnx
import onnxruntime
from tqdm.auto import tqdm


class ONNXModel():
    def __init__(self, onnx_path):
        onnx.checker.check_model(onnx.load(onnx_path))

        self.onnx_session = onnxruntime.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
        self.input_name = self.get_input_name(self.onnx_session)
        self.output_name = self.get_output_name(self.onnx_session)
        self.input_shape = self.get_input_shape()
        self.output_shape = self.get_output_shape()
        self.input_dtype = self.get_intput_dtype()
        self.output_dtype = self.get_output_dtype()

        print(f"{'Inputs':=^45}\n")
        for name, shape in self.input_shape.items():
            print('{0:<15}{1:<15}{2:<15}\n'.format(str(name), str(shape), str(self.input_dtype[name])))

        print(f"{'Outputs':=^45}\n")
        for name, shape in self.output_shape.items():
            print('{0:<15}{1:<15}{2:<15}\n'.format(str(name), str(shape), str(self.output_dtype[name])))
 
    def get_output_name(self, onnx_session):
        output_name = []
        for node in onnx_session.get_outputs():
            output_name.append(node.name)
        return output_name
 
    def get_input_name(self, onnx_session):
        input_name = []
        for node in onnx_session.get_inputs():
            input_name.append(node.name)
        return input_name

    def get_input_shape(self):
        input_shape = {}
        for index, name in enumerate(self.input_name):
            input_shape[name] = self.onnx_session.get_inputs()[index].shape
        return input_shape

    def get_output_shape(self):
        output_shape = {}
        for index, name in enumerate(self.output_name):
            output_shape[name] = self.onnx_session.get_outputs()[index].shape
        return output_shape

    def get_intput_dtype(self):
        input_dtype = {}
        for index, name in enumerate(self.input_name):
            input_dtype[name] = self.onnx_session.get_inputs()[index].type
        return input_dtype

    def get_output_dtype(self):
        output_dtype = {}
        for index, name in enumerate(self.output_name):
            output_dtype[name] = self.onnx_session.get_outputs()[index].type
        return output_dtype
 
    def forward(self, input_feed=None):
        outputs = self.onnx_session.run(self.output_name, input_feed=input_feed)
        return dict(zip(self.output_name, outputs))


if __name__ == '__main__':
    print(onnxruntime.get_device())
    onnx_path = './model.onnx'
    onnx_model = ONNXModel(onnx_path)
    
    # input_feed = {}
    # input_feed["wave_input"] = np.random.rand(1, 80000).astype(np.float32)
    # input_feed["text_input"] = np.random.randint(5, size=(1, 10))
    # outputs = onnx_model.forward(input_feed)
    # print(outputs)

    # cost_list = []
    # epoch = 1000

    # for i in tqdm(range(epoch)):
    #     start_time = time.time()
    #     outputs = onnx_model.forward(input_feed)
    #     time_cost = (time.time() - start_time) * 1000
    #     cost_list.append(time_cost)

    # tp90 = sorted(cost_list)[int(epoch*0.90)]
    # tp95 = sorted(cost_list)[int(epoch*0.95)]
    # tp99 = sorted(cost_list)[int(epoch*0.99)]
    # tp100 = sorted(cost_list)[-1]
    # avg = sum(cost_list) / epoch
    # print(f"Average time {avg}ms")
    # print(f"TP90 time {tp90}ms")
    # print(f"TP95 time {tp95}ms")
    # print(f"TP99 time {tp99}ms")
    # print(f"TP100 time {tp100}ms")

    