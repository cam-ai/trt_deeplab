import tensorrt as trt

G_LOGGER = trt.Logger()

batch_size = 1
imput_h = 612
imput_w = 816

explicit_batch = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)

def get_engine(onnx_model_name, engine_file_path):

    # with trt.Builder(G_LOGGER) as builder, builder.create_network(explicit_batch) as network, builder.create_builder_config() as config, trt.OnnxParser(network, G_LOGGER) as parser:
    with trt.Builder(G_LOGGER) as builder, builder.create_network(
            explicit_batch
    ) as network,builder.create_builder_config() as config, trt.OnnxParser(network, G_LOGGER
    ) as parser, trt.Runtime(G_LOGGER
    ) as runtime:


        config.max_workspace_size = 1 << 32
        # fp16
        config.flags  = 1<< int(trt.BuilderFlag.FP16)
        print('Loading ONNX file from path {}...'.format(onnx_model_name))
        with open(onnx_model_name, 'rb') as model:
            print('Beginning ONNX file parsing')
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))

        print('Completed parsing of ONNX file')
        print('Building an engine from file {}; this may take a while...'.format(onnx_model_name))


        print("num layers:", network.num_layers)
        network.get_input(0).shape = [batch_size, 3, imput_h, imput_w]
        plan = builder.build_serialized_network(network, config)
        engine = runtime.deserialize_cuda_engine(plan)
        print("Completed creating Engine")
        with open(engine_file_path, "wb") as f:
            f.write(plan)
        return engine



def main():
    onnx_file_path = '/media/myue/AHS/框架/MNN/build/deeplab.onnx'
    engine_file_path = './DeepLabV3_F16.trt'

    get_engine(onnx_file_path, engine_file_path)


if __name__ == '__main__':
    main()