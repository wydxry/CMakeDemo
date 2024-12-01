#include <iostream>
#include <opencv2/opencv.hpp>

int main() {
    std::string image_path = "C:\\code\\C++\\demo2\\images\\test.jpg";
    cv::Mat image = cv::imread(image_path);

    if (image.empty()) {
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return -1;
    }

    cv::namedWindow("Display Window", cv::WINDOW_NORMAL);
    cv::imshow("Display Window", image);

    cv::waitKey(0);
	
	system("pause");

    return 0;
}
