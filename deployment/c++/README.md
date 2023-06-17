# HOW TO USE
## requirements
### necessary
>* CUDA Toolkit Documentation v12.0: https://docs.nvidia.com/cuda/archive/12.0.0/cuda-installation-guide-linux/index.html  
>* cuDNN v8.8.0 for CUDA 12.0:https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html  
>* TensorRT 8.6.0 EA: https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-860-ea/quick-start-guide/index.html    

### optional: python 
>* pycuda 2022.2.2
>* tensorrt 8.6.0
>* numpy 1.24.2
>* torch 1.10.2
>* onnx 1.14.0

## quick start

### c++
>* `git clone https://github.com/RLE-Foundation/rllte`  
>* `cd path_to_rllte/deloyment/c++`  
>* `mkdir build && cd build`
>* `cmake .. && make`
>* `./DeployerTest ../../model/test_model.onnx`  

### python
>* `git clone https://github.com/RLE-Foundation/rllte`  
>* `cd path_to_rllte/deloyment/python`
>* `python3 pth2onnx.py ../model/test_model.pth`
>* `./trtexec --onnx=test_model.onnx --saveEngine=test_model.trt --skipInference`
>* `python3 infer.py test_model.plan`

## use in your c++ project
>* `#inlude "RLLTEDeployer.h"`  
    Including the header file in your cpp file.
>* `Options options;`  
    `options.deviceIndex = 0;`  
    `options.doesSupportDynamicBatchSize = false;`  
    `options.maxWorkspaceSize = 4000000000;`  
    `options.precision = Precision::FP16;`  
    Declear an instance of Options, and configurate the parameters.
>* `RLLTEDeployer deployer(options);`  
    Declear an instance of RLLTEDeployer.  
>* `deployer.build(path_of_onnx_model)`  
    Use the build member function to convert the onnx model to the tensorrt static model(plan).
>* `deployer.loadPlan()`  
    `deployer.loadPlan(path_of_the_tensortrt_plan)`
    Use the loadPlan member function to load the converted model. If a path is given, then it will search the path, or it will just search the current working directory.
>* `deployer.infer<float>(input, output, 1);`  
   `deployer.infer<float16_t>(input, output, 1);`  
   `deployer.infer<int8>(input, output, 1);`  
   Use infer member funtion to execute the infer process. The input is the tensor with relevant data type, and the output is a pointer with relevant data size and data type. The infer result will be moved to the output.
>* The complete code please refer to the DeployerTest.cpp;

## c++ project with cmake
>`find_package(CUDA REQUIRED)`  
`include_directories(${CUDA_INCLUDE_DIRS} ${Path_of_RLLTEDeployer_h}})`   
`target_link_libraries(YOUREXECUTEFILE ${PATH_OF_libRLLTEDeployer_so)`  

