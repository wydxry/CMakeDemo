# Steps

```shell
cd demo2
mkdir build
cd build
cmake ..
cmake --build . -- /p:Configuration=Release
cd Release
opencv_win_project.exe
```
![](https://github.com/wydxry/CMakeDemo/blob/main/demo2/assets/demo.jpg)

ps:
opencv path should be added. such as:
```shell
C:\opencv\opencv\build\x64\vc16\bin
```
