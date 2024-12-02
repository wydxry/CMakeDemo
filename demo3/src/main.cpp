#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include "NvInferRuntime.h"
#include <cuda_device_runtime_api.h>
#include "cuda_runtime.h"

#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <cassert>
#include <numeric>

using namespace std;
using namespace nvinfer1;

static const int INPUT_H = 128;
static const int INPUT_W = 128;
static const int OUTPUT_SIZE = 30;

const char* INPUT_BLOB_NAME = "input";
const char* OUTPUT_BLOB_NAME = "output";

int input_data_length = 1 * 2 * 128 * 128;
int output_data_length = 1 * 30;

//构建Logger
class Logger : public ILogger
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} gLogger;

void inferTensorRT()
{
    std::cout << "inferTensorRT begin" << std::endl;
    std::string enginepath = "C:/code/C++/demo3/engine/ResNet10_FP16.engine";	    //读取二进制文件engine的路径
    
    std::cout << "enginepath: " << enginepath << std::endl;
    std::ifstream file(enginepath, std::ios::binary);		// 以二进制方式打开
    char* trtModelStream = NULL;							// 定义一个字符指针，用于读取engine文件数据
    int size = 0;											// 存储二进制文件字符的数量
    if (file.good()) {
        file.seekg(0, file.end);							//将文件指针移动到文件末尾
        size = file.tellg();								//获取当前文件指针的位置，即文件的大小
        file.seekg(0, file.beg);							//文件指针移回文件开始处
        trtModelStream = new char[size];					//分配足够的内存储存文件内容
        assert(trtModelStream);								//检查内存是否分配成功
        file.read(trtModelStream, size);					//读取文件信息，并存储在trtModelStream
        file.close();										//关闭文件
    }
    
    Logger logger;
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
    assert(runtime != nullptr);
    nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trtModelStream, size);
    assert(engine != nullptr);
    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
    assert(context != nullptr);
    delete[] trtModelStream;		

    int nbio = engine->getNbIOTensors();
    const char* inputname = engine->getIOTensorName(0);
    std::cout << "input name :" << inputname << std::endl;
    const char* outputname = engine->getIOTensorName(engine->getNbIOTensors() - 1);
    std::cout << "output name :" << outputname << std::endl;
    Dims input_shape = engine->getTensorShape(inputname);
    Dims output_shape = engine->getTensorShape(outputname);
    auto mInputDims = Dims4(input_shape.d[0], input_shape.d[1], input_shape.d[2], input_shape.d[3]);
    auto mOutputDims = Dims2(output_shape.d[0], output_shape.d[1]);

    std::cout << "inputDims.nbDims: " << mInputDims.nbDims << std::endl;
    std::cout << "outputDims.nbDims: " << mOutputDims.nbDims << std::endl;

    size_t input_size = mInputDims.d[0] * mInputDims.d[1] * mInputDims.d[2] * mInputDims.d[3];
    size_t output_size = mOutputDims.d[0] * mOutputDims.d[1];
    std::cout << "input_size: " << input_size << std::endl;
    std::cout << "output_size: " << output_size << std::endl;
    float* input_buff = (float*)malloc(input_size * sizeof(float));
    float* output_buff = (float*)malloc(output_size * sizeof(float));

    const int height = 128;
    const int width = 128;
    const int batch_size = 1;
    const int img_channels = 2;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < img_channels; j++) {
            for (int k = 0; k < height; k++) {
                for (int l = 0; l < width; l++) {
                    input_buff[i * img_channels * height * width + j * height * width + k * width + l] = (float)1.0 * (rand() % 256);
                }
            }
        }
    }
    void* input_mem{ nullptr };
    void* output_mem{ nullptr };
    try
    {
        cudaMalloc(&input_mem, input_size * sizeof(float));
        cudaMalloc(&output_mem, output_size * sizeof(float));
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }

    context->setTensorAddress(engine->getIOTensorName(0), input_mem);
    context->setTensorAddress(engine->getIOTensorName(engine->getNbIOTensors() - 1), output_mem);

    // Memcpy from host input buffers to device input buffers
    cudaStream_t mStream = 0;
    cudaStreamCreate(&mStream);
    cudaMemcpyAsync(input_mem, input_buff, input_size * sizeof(float), cudaMemcpyHostToDevice, mStream);

    //预热
    for (int i = 0; i < 100; ++i) {
        context->enqueueV3(mStream);
    }

    int epoch = 10000;
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < epoch; ++i) {
        context->enqueueV3(mStream);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_milli = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    auto duration_nano = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "infer time: " << duration_milli * 1.0 / epoch << " ms" << std::endl;
    std::cout << "infer time: " << duration_nano * 1.0 / epoch << " ns" << std::endl;
    std::cout << "fps: " << epoch * 1000.0 / duration_milli << std::endl;
    std::cout << "fps: " << epoch * 1e9 * 1.0 / duration_nano << std::endl;
    std::cout << "infer done." << std::endl;

    // Memcpy from device output buffers to host output buffers
    cudaMemcpyAsync(output_buff, output_mem, output_size * sizeof(float), cudaMemcpyDeviceToHost, mStream);

    for (int i = 0; i < output_size; i++) {
        std::cout << output_buff[i] << "\t";
    }
    std::cout << endl;

    std::cout << "inferTensorRT end" << std::endl;
}

int main(int argc, char** argv)
{
    inferTensorRT();
    getchar();
    return 0;
}
