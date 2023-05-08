#include "HsuanwuDeployer.h"

using namespace nvinfer1;

void Logger::log(Severity severity, const char *msg) noexcept 
{
    if(severity <= Severity::kWARNING) 
    {
        std::cout << msg << std::endl;
    }
}

HsuanwuDeployer::HsuanwuDeployer(const Options& options): m_options(options)
{
    if(m_options.doesSupportDynamicBatchSize == false) 
    {
        std::cout << "Set optBatchSize and maxBatchSize to 1." << std::endl;
        m_options.optBatchSize = 1;
        m_options.maxBatchSize = 1;
    }
    if(m_options.engineName[0] == '0')
    {
        std::cout << "Set engine name to engine." << std::endl;
        strncpy(m_options.engineName, "engine", 6);
    }
    setEngineName();
    std::cout<<"Engine name: "<<m_engineName<<std::endl;
}

void HsuanwuDeployer::checkCudaErrorCode(cudaError_t code) {
    if (code != 0) {
        std::string errMsg = "CUDA failed!\nCode: " + std::to_string(code) + "(" + cudaGetErrorName(code) + ")\nMessage: " + cudaGetErrorString(code);
        std::cout << errMsg << std::endl;
        throw std::runtime_error(errMsg);
    }
}

HsuanwuDeployer::~HsuanwuDeployer()
{
}

void HsuanwuDeployer::setEngineName()
{
    std::string engineName(m_options.engineName);
    std::vector<std::string> deviceNames;
    int numGPUs;
    cudaGetDeviceCount(&numGPUs);
    for (int device=0; device<numGPUs; device++) 
    {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        deviceNames.push_back(std::string(prop.name));
    }
    if(static_cast<size_t>(m_options.deviceIndex) >= deviceNames.size())
    {
        throw std::runtime_error("Error, device index is out of range!");
    }
    auto deviceName = deviceNames[m_options.deviceIndex];
    engineName+= "_" + deviceName;
    if (m_options.precision == Precision::FP16) 
    {
        engineName += "_fp16";
    }else 
    {
        engineName += "_fp32";
    }
    engineName += "_" + std::to_string(m_options.maxBatchSize);
    engineName += "_" + std::to_string(m_options.optBatchSize);
    engineName += "_" + std::to_string(m_options.maxWorkspaceSize);
    engineName += ".plane";
    std::replace(engineName.begin(), engineName.end(), ' ', '_');
    m_engineName = engineName;
}

bool HsuanwuDeployer::checkFile(const std::string &filepath) 
{
    std::ifstream f(filepath.c_str());
    return f.good();
}

bool HsuanwuDeployer::build(const std::string & onnxModelPath) 
{
    std::cout << "Searching for engine file: " << m_engineName << std::endl;

    if(checkFile(m_engineName))
    {
        std::cout << "Found, skip building." << std::endl;
        return true;
    }

    if(checkFile(onnxModelPath) == false) 
    {
        throw std::runtime_error("No onnx model at path: " + onnxModelPath);
    }

    std::cout << "No engine but onnx, building..." << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    builder->setMaxBatchSize(m_options.maxBatchSize);

    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }
    file.close();
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    const auto input = network->getInput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];

    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    config->addOptimizationProfile(optProfile);

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.precision == Precision::FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    plan->destroy();
    config->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    std::ofstream outfile(m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    outfile.close();
    std::cout << "Success, saved engine to " << m_engineName << std::endl;
    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

bool HsuanwuDeployer::build(const std::string & onnxModelPath, const std::string & engineSavePath) 
{
    std::cout << "Searching for engine file: " << engineSavePath+"/"+m_engineName << std::endl;

    if(checkFile(engineSavePath+"/"+m_engineName))
    {
        std::cout << "Found, skip building." << std::endl;
        return true;
    }

    if(checkFile(onnxModelPath) == false) 
    {
        throw std::runtime_error("No onnx model at path: " + onnxModelPath);
    }

    std::cout << "No engine but onnx, building..." << std::endl;

    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(m_logger));
    if (!builder) {
        return false;
    }

    builder->setMaxBatchSize(m_options.maxBatchSize);

    auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network) {
        return false;
    }

    auto parser = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, m_logger));
    if (!parser) {
        return false;
    }

    std::ifstream file(onnxModelPath, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read engine file");
    }
    file.close();
    auto parsed = parser->parse(buffer.data(), buffer.size());
    if (!parsed) {
        return false;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config) {
        return false;
    }

    const auto input = network->getInput(0);
    const auto inputName = input->getName();
    const auto inputDims = input->getDimensions();
    int32_t inputC = inputDims.d[1];
    int32_t inputH = inputDims.d[2];
    int32_t inputW = inputDims.d[3];

    IOptimizationProfile *optProfile = builder->createOptimizationProfile();
    optProfile->setDimensions(inputName, OptProfileSelector::kMIN, Dims4(1, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, OptProfileSelector::kOPT, Dims4(m_options.optBatchSize, inputC, inputH, inputW));
    optProfile->setDimensions(inputName, OptProfileSelector::kMAX, Dims4(m_options.maxBatchSize, inputC, inputH, inputW));
    config->addOptimizationProfile(optProfile);

    config->setMaxWorkspaceSize(m_options.maxWorkspaceSize);

    if (m_options.precision == Precision::FP16) {
        config->setFlag(BuilderFlag::kFP16);
    }

    cudaStream_t profileStream;
    checkCudaErrorCode(cudaStreamCreate(&profileStream));
    config->setProfileStream(profileStream);

    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan) {
        return false;
    }
    plan->destroy();
    config->destroy();
    network->destroy();
    parser->destroy();
    builder->destroy();
    std::ofstream outfile(engineSavePath+"/"+m_engineName, std::ofstream::binary);
    outfile.write(reinterpret_cast<const char*>(plan->data()), plan->size());
    outfile.close();
    std::cout << "Success, saved engine to " << engineSavePath+"/"+m_engineName << std::endl;
    checkCudaErrorCode(cudaStreamDestroy(profileStream));
    return true;
}

bool HsuanwuDeployer::loadPlane(const std::string &planeFile)
{
    std::ifstream file(planeFile, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        throw std::runtime_error("Unable to read plane file");
    }
    file.close();
    m_runtime.reset(nvinfer1::createInferRuntime(m_logger));
    if (!m_runtime) {
        return false;
    }
    cudaError_t ret = cudaSetDevice(m_options.deviceIndex);
    if (ret != 0) {
        int numGPUs;
        cudaGetDeviceCount(&numGPUs);
        std::string errMsg = "Unable to use GPU: " + std::to_string(m_options.deviceIndex)+".";
        throw std::runtime_error(errMsg);
    }
    m_engine.reset(m_runtime->deserializeCudaEngine(buffer.data(), buffer.size()));
    if (!m_engine) {
        return false;
    }
    m_context.reset(m_engine->createExecutionContext());
    if (!m_context) {
        return false;
    }
    nvinfer1::IEngineInspector* engineInspector = m_engine->createEngineInspector();
    std::cout<<engineInspector->getEngineInformation(nvinfer1::LayerInformationFormat::kJSON)<<std::endl;
    int numofbinding = m_engine->getNbBindings();
    for(int i = 0; i< numofbinding; i++)
    {
        if(m_engine->bindingIsInput(i)){
            m_input_indexs.emplace_back(i);
            m_input_dims.emplace_back(m_engine->getBindingDimensions(i));
            std::cout<<"Input binding, index: "<<i<<", name: "<<m_engine->getBindingName(i)<<", format"<<m_engine->getBindingFormatDesc(i)<<std::endl;
            m_input_names.emplace_back(m_engine->getBindingName(i));
            int tmp_size = 1;
            for (int d = 0; d < m_engine->getBindingDimensions(i).nbDims; d ++)
            {
                tmp_size *= m_engine->getBindingDimensions(i).d[d];
                printf("%d,", m_engine->getBindingDimensions(i).d[d]);
            }
            m_input_byte_sizes.emplace_back(tmp_size*sizeof(float));
            printf("\n");
        }else{
            m_output_indexs.emplace_back(i);
            m_output_dims.emplace_back(m_engine->getBindingDimensions(i));
            std::cout<<"Output binding, index: "<<i<<", name: "<<m_engine->getBindingName(i)<<", format"<<m_engine->getBindingFormatDesc(i)<<std::endl;
            m_output_names.emplace_back(m_engine->getBindingName(i));
            int tmp_size = 1;
            for (int d = 0; d < m_engine->getBindingDimensions(i).nbDims; d ++)
            {
                tmp_size *= m_engine->getBindingDimensions(i).d[d];
                printf("%d,", m_engine->getBindingDimensions(i).d[d]);
            }
            m_output_byte_sizes.emplace_back(tmp_size*sizeof(float));
            printf("\n");
        }
    }
    return true;
}

bool HsuanwuDeployer::infer(std::vector<float*> input, std::vector<float*> output, int batchSize)
{
    samplesCommon::BufferManager buffers(m_engine);
    for(int i = 0; i<m_input_names.size();i++)
    {
        memcpy(buffers.getHostBuffer(m_input_names[i]), input[i], m_input_byte_sizes[i]*batchSize);
    }
    buffers.copyInputToDevice();
    bool status = m_context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }
    buffers.copyOutputToHost();
    for(int i = 0; i<m_output_names.size();i++)
    {
        memcpy(output[i], buffers.getHostBuffer(m_output_names[i]), m_output_byte_sizes[i]*batchSize);
    }
    return true;
}