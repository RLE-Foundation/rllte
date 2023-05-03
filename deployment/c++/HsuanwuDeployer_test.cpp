#include "HsuanwuDeployer.h"

int main(int argc, char** argv)
{
    Options options;
    options.deviceIndex = 0;
    options.doesSupportDynamicBatchSize = false;
    options.maxWorkspaceSize = 4000000000;
    options.precision = Precision::FP32;
    HsuanwuDeployer hsuanwuDeployer(options);
    if (!hsuanwuDeployer.build("/home/jakeluo/Documents/Hsuanwu/deployment/python/encoder.onnx", "/home/jakeluo/Documents/Hsuanwu/deployment/c++/"))
    {
        throw std::runtime_error("Unable to build plane.");
    }

    if (!hsuanwuDeployer.loadPlane("/home/jakeluo/Documents/Hsuanwu/deployment/c++/engine_NVIDIA_GeForce_GTX_1080_Ti_fp32_1_1_4000000000.plane"))
    {
        throw std::runtime_error("Unable to load plane.");
    }
    float input[9*84*84];
    for(int i = 0; i < 9; i++)
    {
        for(int j = 0; j < 84; j++)
        {
            for(int k = 0; k<84;k++)
            {
                input[i+j*9+k] = 1.0f;
            }
        }
    }
    float output[50];
    hsuanwuDeployer.infer(input, output, 1);
    std::cout<<"infer end."<<std::endl;
    return 0;
}