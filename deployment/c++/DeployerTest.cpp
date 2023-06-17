#include "RLLTEDeployer.h"
#include <chrono>
#include <thread>
int main(int argc, char** argv)
{
    Options options;
    options.deviceIndex = 0; 
    options.doesSupportDynamicBatchSize = false;
    options.maxWorkspaceSize = 4000000000;
    options.precision = Precision::FP16;
    RLLTEDeployer deployer(options);
    if (!deployer.build(argv[1]))
    {
        throw std::runtime_error("Unable to build plane.");;
    }

    if (!deployer.loadPlan())
    {
        throw std::runtime_error("Unable to load plane.");
    }
    std::vector<float16_t*> input;;
    float16_t* input_tmp, *output_tmp;
    input.emplace_back(input_tmp);
    for(int i = 0; i < input.size(); i++)
    {
        input[i] = new float16_t[9*84*84];
        std::fill_n(input[i], 9*84*84, 3.0);
    }
    std::vector<float16_t*> output;
    output.emplace_back(output_tmp);
    for(int i = 0; i < output.size(); i++)
    {
        output[i] = new float16_t[50];
    }
    
    deployer.infer<float16_t>(input, output, 1);
    std::cout<<" infer end."<<std::endl;
    for(int i = 0; i < output.size(); i++)
    {
        for(int j = 0; j< 50; j++)std::cout<<output[i][j]<<" ";
        std::cout<<"\n";
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