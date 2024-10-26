#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    //Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    std::cout << "ONNX Runtime initialized." << std::endl;

    return 0;
}