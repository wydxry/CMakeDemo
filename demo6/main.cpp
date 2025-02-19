#include <QCoreApplication>
#include <iostream>
#include <hidapi.h>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    std::cout << "hid begin" << std::endl;

    // 初始化HIDAPI
    if (hid_init() != 0) {
        std::cerr << "Failed to initialize HIDAPI" << std::endl;
        return 0;
    }

    // 获取所有HID设备
    struct hid_device_info *devs, *cur_dev;
    devs = hid_enumerate(0x0, 0x0); // 0x0表示列出所有设备

    cur_dev = devs;
    while (cur_dev) {
        std::wcout << L"Device Found: " << cur_dev->path << std::endl;
        std::wcout << L"  Manufacturer: " << cur_dev->manufacturer_string << std::endl;
        std::wcout << L"  Product: " << cur_dev->product_string << std::endl;
        std::wcout << L"  VID: " << std::hex << cur_dev->vendor_id << std::dec << std::endl;
        std::wcout << L"  PID: " << std::hex << cur_dev->product_id << std::dec << std::endl;
        std::wcout << L"  Release Number: " << cur_dev->release_number << std::endl;
        std::wcout << L"  Interface Number: " << cur_dev->interface_number << std::endl;
        std::wcout << L"  Usage Page: " << cur_dev->usage_page << std::endl;
        std::wcout << L"  Usage: " << cur_dev->usage << std::endl;
        std::wcout << std::endl;

        cur_dev = cur_dev->next;

        break;
    }

    // 释放设备信息
    hid_free_enumeration(devs);

    // 关闭HIDAPI
    hid_exit();

    std::cout << "hid end" << std::endl;

    return a.exec();
}
