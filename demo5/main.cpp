#include <QCoreApplication>
#include <iostream>
#include <Eigen/Dense>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::cout << "test" << std::endl;

    // 定义两个3x3的矩阵
    Eigen::Matrix3d A;
    Eigen::Matrix3d B;

    A << 1, 2, 3,
        4, 5, 6,
        7, 8, 9;

    B << 9, 8, 7,
        6, 5, 4,
        3, 2, 1;

    // 矩阵相加
    Eigen::Matrix3d C = A + B;

    // 输出结果
    std::cout << "Matrix A:\n" << A << "\n\n";
    std::cout << "Matrix B:\n" << B << "\n\n";
    std::cout << "Matrix C = A + B:\n" << C << "\n";

    return a.exec();
}
