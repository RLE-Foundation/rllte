#pragma once

#include "NvInfer.h"
#include "NvOnnxParser.h"
#include <iostream>
#include <vector>
#include <memory>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <thread>
#include "common/buffers.h"
#include "cuda_fp16.h"

typedef half float16_t;

enum class Precision {
    FP32,
    FP16,
    INT8
};

struct Options {
    char engineName[100] = {'0'};
    bool doesSupportDynamicBatchSize = true;
    Precision precision = Precision::FP16;
    int32_t optBatchSize = 1;
    int32_t maxBatchSize = 16;
    size_t maxWorkspaceSize = 4000000000;
    int deviceIndex = 0;
};

class Logger : public nvinfer1::ILogger {
    void log (Severity severity, const char* msg) noexcept override;
};

class RLLTEDeployer{
public:
    RLLTEDeployer(const Options& options);
    ~RLLTEDeployer();
    bool build(const std::string & onnxModelPath, const std::string & engineSavePath);
    bool build(const std::string & onnxModelPath);
    bool loadPlan(const std::string &planFile);
    bool loadPlan();
    template<class T>
    bool infer(std::vector<T*> input, std::vector<T*> output, int batchSize);

private:
    void setEngineName();

    void getDeviceNames(std::vector<std::string>& deviceNames);

    bool checkFile(const std::string& filepath);

    std::vector<void*> m_buffers;
    std::vector<uint32_t> m_outputLengthsFloat{};

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine = nullptr;
    std::shared_ptr<nvinfer1::IExecutionContext> m_context = nullptr;
    std::shared_ptr<nvinfer1::IRuntime> m_runtime = nullptr;
    Options m_options;
    Logger m_logger;
    std::string m_engineName;
    std::vector<nvinfer1::Dims> m_input_dims;
    std::vector<nvinfer1::Dims> m_output_dims;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<int> m_input_T_sizes;
    std::vector<int> m_output_T_sizes;
    std::vector<int> m_input_indexs;
    std::vector<int> m_output_indexs;

    inline void checkCudaErrorCode(cudaError_t code);
};
