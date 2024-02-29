#include "ncnn/net.h"

int main(int argc, char** argv)
{
    ncnn::Net net; //net 
    net.load_param(argv[1]);//load the param file
    net.load_model(argv[2]);//load the bin file


    ncnn::Mat in;//data tyoe Mat. Input/output data are stored in this structure. 
    in.create(84 ,84, 9);//create a 9*84*84 tensor as the test onnx model requires. 
    in.fill(3.0f);//fill the input data will 3.0.


    ncnn::Extractor ex = net.create_extractor();//create a extractor from net 
    ex.set_light_mode(true);
    ex.set_num_threads(4);
    ex.input("input", in);//feed data into the extractor, then it will start infer automatically.
    ncnn::Mat out;
    ex.extract("output", out);//get infer result and put it into the out variable.


    for (int q=0; q<out.c; q++)//print the infer result
    {
        const float* ptr = out.channel(q);
        for (int z=0; z<out.d; z++)
        {
            for (int y=0; y<out.h; y++)
            {
                for (int x=0; x<out.w; x++)
                {
                    printf("%f ", ptr[x]);
                }
                ptr += out.w;
                printf("\n");
            }
            printf("\n");
        }
        printf("------------------------\n");
    }


    ex.clear();//destrcut the extractor
    net.clear();//destrcut the net
    return 0;
}
