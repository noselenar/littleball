#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    // Initialize ONNX Runtime
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXModel");
    std::cout << "ONNX Runtime initialized." << std::endl;

    try {
        // Create session options without GPU
        Ort::SessionOptions session_options;

        // Create ONNX Runtime session
        Ort::Session session(env, L"best_1.onnx", session_options);

        // Read and preprocess image
        cv::Mat image = cv::imread("./input.jpg");
        if (image.empty()) {
            std::cerr << "Error: Could not read image." << std::endl;
            return -1;
        }

        Mat resized_image;
        resize(image, resized_image, Size(1920, 1920));  // Adjust according to your model input size
        resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);  // Normalize to [0, 1]

        // Create input tensor
        std::vector<int64_t> input_shape = {1, 3, resized_image.rows, resized_image.cols};  // CHW format
        std::vector<float> input_tensor_values(resized_image.total() * resized_image.channels());
        memcpy(input_tensor_values.data(), resized_image.data, input_tensor_values.size() * sizeof(float));

        // Use Ort::MemoryInfo to create the tensor
        Ort::AllocatorWithDefaultOptions allocator;
        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_tensor_values.data(), input_tensor_values.size(), input_shape.data(), input_shape.size()
        );

        // Set model input names
        const char* input_names[] = {"images"};  // Adjust input name if necessary
        const char* output_names[] = {"output0"};  // Assuming the model output name is "output"

        // Perform inference
        auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor, 1, output_names, 1);
        float* output_data = output_tensors.front().GetTensorMutableData<float>();

        // Define number of detections
        const int NUM_DETECTIONS = 6;  // Adjust based on your model output


        for (int i = 0; i < NUM_DETECTIONS * 10; ++i) {
            cout << "Output data[" << i << "] = " << output_data[i] << endl;
        }


        // Post-process output
        for (int i = 0; i < NUM_DETECTIONS; ++i) {
            float confidence = output_data[i * 5 + 4];  // Assuming confidence is the 5th element
            if (confidence > 0) {  // Set threshold
                float x_center = output_data[i * 5] * image.cols;  // 中心 x 坐标
                float y_center = output_data[i * 5 + 1] * image.rows;  // 中心 y 坐标
                float width = output_data[i * 5 + 2] * image.cols;  // 宽度
                float height = output_data[i * 5 + 3] * image.rows;  // 高度

                // 计算左上角坐标
                int x = static_cast<int>(x_center - width / 2);
                int y = static_cast<int>(y_center - height / 2);

                // 绘制检测框
                rectangle(image, Point(x, y), Point(x + static_cast<int>(width), y + static_cast<int>(height)), Scalar(0, 255, 0), 2);
                cout << "Detected object at: (" << x << ", " << y << ", " << width << ", " << height << ")" << endl;
            }
        }

        // Show results
        //imshow("Detections", image);
        imwrite("output.jpg", image);

    } catch (const Ort::Exception& e) {
        std::cerr << "ONNX Runtime error: " << e.what() << std::endl;
        return -1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
