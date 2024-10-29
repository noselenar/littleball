#include <iostream>
#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>

using namespace cv;
using namespace std;

int main() {
    std::cout << "OpenCV version: " << CV_VERSION << std::endl;

    cv::Mat image = cv::imread("./input.jpg");
    imshow("image", image);

    waitKey(0);

    return 0;
}
