# 导入必用依赖
import tensorrt as trt
import pycuda.autoinit  #负责数据初始化，内存管理，销毁等
import pycuda.driver as cuda  #GPU CPU之间的数据传输
import numpy as np
import sys
# 创建logger：日志记录器
logger = trt.Logger(trt.Logger.WARNING)
# 创建runtime并反序列化生成engine
with open(sys.argv[1], 'rb') as f, trt.Runtime(logger) as runtime:
    engine = runtime.deserialize_cuda_engine(f.read())
# 分配CPU锁页内存和GPU显存
with engine.create_execution_context() as context:
    # Set input shape based on image dimensions for inference
    context.set_binding_shape(engine.get_binding_index("input"), (1, 9, 84, 84))
    # Allocate host and device buffers
    bindings = []
    for binding in engine:
        binding_idx = engine.get_binding_index(binding)
        size = trt.volume(context.get_binding_shape(binding_idx))
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        if engine.binding_is_input(binding):
            data = np.random.rand(9,84,84)
            input_buffer = np.ascontiguousarray(data, dtype=np.float16)
            input_memory = cuda.mem_alloc(data.nbytes)
            bindings.append(int(input_memory))
        else:
            output_buffer = cuda.pagelocked_empty(size, dtype)
            output_memory = cuda.mem_alloc(output_buffer.nbytes)
            bindings.append(int(output_memory))

stream = cuda.Stream()
# Transfer input data to the GPU.
cuda.memcpy_htod_async(input_memory, input_buffer, stream)
# Run inference
context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
# Transfer prediction output from the GPU.
cuda.memcpy_dtoh_async(output_buffer, output_memory, stream)
# Synchronize the stream
stream.synchronize()
print(output_buffer)