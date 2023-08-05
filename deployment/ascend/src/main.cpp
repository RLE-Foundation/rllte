#include "acl/acl.h"
#include <iostream>
#include <fstream>
#include <cstring>
#include <map>

using namespace std;

int32_t deviceId = 0;
void InitResource()
{
	aclError ret = aclInit(nullptr);
	ret = aclrtSetDevice(deviceId);
}

uint32_t modelId;
void LoadModel(const char* modelPath)
{
	aclError ret = aclmdlLoadFromFile(modelPath, &modelId);
}

size_t DataSize = 0;
void *HostData;
void *DeviceData;

// Allocate memory on host and input data. 
void PrepareDataTotHost()
{
    HostData = new float[9*84*84];
    std::fill_n(HostData, 9*84*84, 3.0);
    DataSize = 9*84*84;
}

// Allocate device memory and transfer the data in the memory to the device through memory copy.
void CopyDataFromHostToDevice()
{
	aclError ret = aclrtMalloc(&DeviceData, DataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	ret = aclrtMemcpy(DeviceData, DataSize, HostData, DataSize, ACL_MEMCPY_HOST_TO_DEVICE);
}

void LoadData()
{
	PrepareDataTotHost()
	CopyDataFromHostToDevice();
}

aclmdlDataset *inputDataSet;
aclDataBuffer *inputDataBuffer;
aclmdlDataset *outputDataSet;
aclDataBuffer *outputDataBuffer;
aclmdlDesc *modelDesc;
size_t outputDataSize = 0;
void *outputDeviceData;

// Prepare the input data structure for model inference.
void CreateModelInput()
{
        // Create data of the aclmdlDataset type to describe the input for model inference.
	inputDataSet = aclmdlCreateDataset();
	inputDataBuffer = aclCreateDataBuffer(DeviceData, DataSize);
	aclError ret = aclmdlAddDatasetBuffer(inputDataSet, inputDataBuffer);
}

// Prepare the output data structure for model inference.
void CreateModelOutput()
{
       // Create model description.
	modelDesc =  aclmdlCreateDesc();
	aclError ret = aclmdlGetDesc(modelDesc, modelId);
        // Create data of the aclmdlDataset type to describe the output for model inference.
	outputDataSet = aclmdlCreateDataset();
        // Obtain the buffer size (in bytes) occupied by the model output data.
	outputDataSize = 50;
        // Allocate output buffer.
	ret = aclrtMalloc(&outputDeviceData, outputDataSize, ACL_MEM_MALLOC_HUGE_FIRST);
	outputDataBuffer = aclCreateDataBuffer(outputDeviceData, outputDataSize);
	ret = aclmdlAddDatasetBuffer(outputDataSet, outputDataBuffer);
}

// Execute the model.
void Inference()
{
    CreateModelInput();
	CreateModelOutput();
	aclError ret = aclmdlExecute(modelId, inputDataSet, outputDataSet);
}

void *outputHostData;

void PrintResult()
{
        // Obtain the inference result data.
	aclError ret = aclrtMallocHost(&outputHostData, outputDataSize);
	ret = aclrtMemcpy(outputHostData, outputDataSize, outputDeviceData, outputDataSize, ACL_MEMCPY_DEVICE_TO_HOST);
        // Cast the buffered data to the float type.
	float* outFloatData = reinterpret_cast<float *>(outputHostData);
	
    std::cout<<" infer end."<<std::endl;
    for(int j = 0; j< 50; j++)std::cout<<*(outFloatData+i)<<" ";
    std::cout<<"\n";
}

void UnloadModel()
{
    // Destroy the model description.
	aclmdlDestroyDesc(modelDesc);
    // Unload the model.
	aclmdlUnload(modelId);
}

void Unloadata()
{
	aclError ret = aclrtFreeHost(HostData);
	HostData = nullptr;
	ret = aclrtFree(DeviceData);
	DeviceData = nullptr;
	aclDestroyDataBuffer(inputDataBuffer);
	inputDataBuffer = nullptr;
	aclmdlDestroyDataset(inputDataSet);
	inputDataSet = nullptr;
	
	ret = aclrtFreeHost(outputHostData);
	outputHostData = nullptr;
	ret = aclrtFree(outputDeviceData);
	outputDeviceData = nullptr;
	aclDestroyDataBuffer(outputDataBuffer);
	outputDataBuffer = nullptr;
	aclmdlDestroyDataset(outputDataSet);
	outputDataSet = nullptr;
}

void DestroyResource()
{
	aclError ret = aclrtResetDevice(deviceId);
	aclFinalize();
}

int main()
{	
    // 1. Define a resource initialization function for AscendCL initialization and runtime resource allocation (specifying a compute device).
	InitResource();
	
    // 2. Define a model loading function for loading the image classification model.
	const char *modelPath = "../model/test_model.om";
	LoadModel(modelPath);
	
    // 3. Define a function for prepare data to the memory and transferring the data to the device.
    LoadData()
	
    // 4. Define an inference function for executing inference.
	Inference();
	
    // 5. Define a function for processing inference result data to print the class indexes of the top 5 confidence values of the test image.
	PrintResult();
	
    // 6. Define a function for unloading the image classification model.
	UnloadModel();
	
    // 7. Define a function for freeing the memory and destroying inference-related data to prevent memory leak.
	UnloadData();
	
    // 8. Define a resource deinitialization function for AscendCL deinitialization and runtime resource deallocation (releasing a compute device).
	DestroyResource();
}