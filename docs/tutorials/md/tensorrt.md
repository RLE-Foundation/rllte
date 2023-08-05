# with TensorRT

## Prerequisites
Get the complete repository from GitHub:
``` sh
git clone https://github.com/RLE-Foundation/rllte
```

Download the following necessary libraries:

- [CUDA Toolkit Documentation v12.0](https://docs.nvidia.com/cuda/archive/12.0.0/cuda-installation-guide-linux/index.html)
- [cuDNN v8.8.0 for CUDA 12.0](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html)
- [TensorRT 8.6.0 EA](https://docs.nvidia.com/deeplearning/tensorrt/archives/tensorrt-860-ea/quick-start-guide/index.html)

Meanwhile, install the following Python packages:

- pycuda==2022.2.2
- tensorrt==8.6.0
- numpy==1.24.2
- torch==2.0.0
- onnx==1.14.0

The following two examples can used to verify your installation:

``` sh title="C++ Port"
cd deloyment/c++
mkdir build && cd build
cmake .. && make
./DeployerTest ../../model/test_model.onnx
```

``` sh title="Python Port"
cd deloyment/python
python3 pth2onnx.py ../model/test_model.pth
./trtexec --onnx=test_model.onnx --saveEngine=test_model.trt --skipInference
python3 infer.py test_model.plan
```

## Use in Your C++ Project
The following code illustrates how to include our library in you project:
``` c++ title="example.cpp"
// Including the header file in your cpp file.
#inlude "RLLTEDeployer.h

// Declear an instance of Options, and configurate the parameters.
Options options;
options.deviceIndex = 0;  
options.doesSupportDynamicBatchSize = false;  
options.maxWorkspaceSize = 4000000000; 
options.precision = Precision::FP16;

// Declear an instance of Options, and configurate the parameters.
RLLTEDeployer deployer(options);

// Use the build member function to convert the onnx model to the TensorRT static model (plan).
deployer.build(path_of_onnx_model);

// Use the loadPlan member function to load the converted model. If a path is given, 
// then it will search the path, or it will just search the current working directory.
deployer.loadPlan();

// Use infer member funtion to execute the infer process. The input is the tensor with 
// relevant data type, and the output is a pointer with relevant data size and data type. T
// he infer result will be moved to the output.
deployer.infer<float>(input, output, 1);
deployer.infer<float16_t>(input, output, 1);
deployer.infer<int8>(input, output, 1);

...
```
Please refer to the [DeployerTest.cpp](https://github.com/RLE-Foundation/rllte/blob/main/deployment/c%2B%2B/DeployerTest.cpp) for the complete code.

### with `CMake`
``` txt title="CMakeLists.txt"
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS} ${Path_of_RLLTEDeployer_h}})
target_link_libraries(YOUREXECUTEFILE ${PATH_OF_libRLLTEDeployer_so)
```

### with `Docker`
Install the NVIDIA docker via (make sure the NVIDIA driver is installed):
``` sh title="install_docker.sh"
sudo apt-get install ca-certificates gnupg lsb-release
sudo mkdir -p /etc/apt/keyrings

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
sudo groupadd docker
sudo gpasswd -a $USER docker
```

**Restart** your device, and run the following command.
``` sh
sudo service docker restart
```

Now you can run your model via:
``` sh title="run_docker.sh"
docker pull jakeshihaoluo/rllte_deployment_env:0.0.1
docker run -it -v ${path_to_the_repo}:/rllte --gpus all jakeshihaoluo/rllte_deployment_env:0.0.1
cd /rllte/deloyment/c++
mkdir build && cd build
cmake .. && make
./DeployerTest ../../model/test_model.onnx
```