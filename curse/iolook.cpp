 
#include <iostream>
#include<onnxruntime_cxx_api.h>
using namespace std;
using namespace Ort;
int main()
{
 
    const wchar_t* model_path = L"best.onnx";//模型路径
    Ort::Env env;//创建env
    Ort::Session session(nullptr);//创建一个空会话
    Ort::SessionOptions sessionOptions{ nullptr };//创建会话配置
    session = Ort::Session(env, model_path, sessionOptions);
 
    //获取输入节点数量，名称和shape
 
    size_t inputNodeCount= session.GetInputCount();
    std::cout << "input nums:" << inputNodeCount << "\n";
 
    Ort::AllocatorWithDefaultOptions allocator;
 
    std::shared_ptr<char> inputName = std::move(session.GetInputNameAllocated(0, allocator));
    std::vector<char*> inputNodeNames;
    inputNodeNames.push_back(inputName.get());
    std::cout << "input name:" << inputName << "\n";
 
 
    Ort::TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputNodeDataType = input_tensor_info.GetElementType();
    std::vector<int64_t> inputTensorShape = input_tensor_info.GetShape();
    std::cout << "input shape:";
    for (int i = 0; i<inputTensorShape.size(); i++)
    {
        std::cout << inputTensorShape[i]<<" ";
    }
    std::cout << "\n";
 
 
    //获取输出节点数量、名称和shape
    size_t outputNodeCount = session.GetOutputCount();
    std::cout << "output nums:" << outputNodeCount << "\n";
 
    std::shared_ptr<char> outputName = std::move(session.GetOutputNameAllocated(0, allocator));
    std::vector<char*> outputNodeNames;
    outputNodeNames.push_back(outputName.get());
    std::cout << "output names:" << outputName << "\n";
 
 
    Ort::TypeInfo type_info_output0(nullptr);
    type_info_output0 = session.GetOutputTypeInfo(0);  //output0
 
    auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputNodeDataType = tensor_info_output0.GetElementType();
    std::vector<int64_t> outputTensorShape = tensor_info_output0.GetShape();
    std::cout << "output shape:";
    for (int i = 0; i<outputTensorShape.size(); i++)
    {
        std::cout << outputTensorShape[i]<<" ";
    }
    std::cout << "\n";
	getchar();
}