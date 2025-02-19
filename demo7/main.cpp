#include <QCoreApplication>
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // Set up code that uses the Qt event loop here.
    // Call a.quit() or a.exit() to quit the application.
    // A not very useful example would be including
    // #include <QTimer>
    // near the top of the file and calling
    // QTimer::singleShot(5000, &a, &QCoreApplication::quit);
    // which quits the application after 5 seconds.

    // If you do not need a running Qt event loop, remove the call
    // to a.exec() or use the Non-Qt Plain C++ Application template.

    std::cout << "test" << std::endl;

    // 读取一张图片
    cv::Mat image = cv::imread("xxx.jpg");

    if (image.empty()) {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    // 显示图片
    cv::namedWindow("Test OpenCV", cv::WINDOW_AUTOSIZE);
    cv::imshow("Test OpenCV", image);

    // 等待按键
    cv::waitKey(0);

    std::cout << "opencv end" << std::endl;

    return a.exec();
}
