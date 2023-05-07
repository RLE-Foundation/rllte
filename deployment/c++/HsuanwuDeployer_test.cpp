#include "HsuanwuDeployer.h"
#include <chrono>
#include <thread>
int main(int argc, char** argv)
{
    Options options;
    options.deviceIndex = 0;
    options.doesSupportDynamicBatchSize = false;
    options.maxWorkspaceSize = 4000000000;
    options.precision = Precision::INT8;
    HsuanwuDeployer hsuanwuDeployer(options);
    if (!hsuanwuDeployer.build("/home/jakeluo/Documents/Hsuanwu/deployment/python/encoder.onnx", "/home/jakeluo/Documents/Hsuanwu/deployment/c++/"))
    {
        throw std::runtime_error("Unable to build plane.");
    }

    if (!hsuanwuDeployer.loadPlan("/home/jakeluo/Documents/Hsuanwu/deployment/c++/engine_NVIDIA_GeForce_GTX_1080_Ti_int8_1_1_4000000000.plan"))
    {
        throw std::runtime_error("Unable to load plane.");
    }
    std::vector<uint8_t*> input;
    uint8_t* input_tmp, *output_tmp;
    input.emplace_back(input_tmp);
    for(int i = 0; i < input.size(); i++)
    {
        input[i] = new uint8_t[9*84*84];
        std::fill_n(input[i], 9*84*84, 3);
    }
    std::vector<uint8_t*> output;
    output.emplace_back(output_tmp);
    for(int i = 0; i < output.size(); i++)
    {
        output[i] = new uint8_t[50];
    }
    
    for(int i = 0; i < 2; i++)
    {
        hsuanwuDeployer.infer<uint8_t>(input, output, 1);
        std::cout<< i <<" infer end."<<std::endl;
        for(int i = 0; i < output.size(); i++)
        {
            for(int j = 0; j< 50; j++)std::cout<<(float)output[i][j]<<" ";
            std::cout<<"\n";
        }
    }
    for(int i = 0; i < input.size(); i++)
    {
        delete [] input[i];
    }
    for(int i = 0; i < output.size(); i++)
    {
        delete [] output[i];
    }
    return 0;
}