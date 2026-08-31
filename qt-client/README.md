# SAM 3D Qt 客户端

这是一个独立的 Qt Widgets + OpenGL 桌面客户端，按 `design` 中的 1280 × 800 设计稿实现显微图像选区、积分确认、生成状态、失败重试、3D 模型查看与保存流程。

## 构建

项目使用 Qt 5.15.2 `msvc2019_x64`。在 Visual Studio x64 开发环境中执行：

也可以直接双击 `build-release.bat` 构建，再双击 `run-release.bat` 启动。

```bat
mkdir build
cd build
D:\Qt\5.15.2\msvc2019_x64\bin\qmake.exe ..\SAM3DQtClient.pro CONFIG+=release
nmake
```

运行时请将 `D:\Qt\5.15.2\msvc2019_x64\bin` 加入 `PATH`，或使用同目录下的 `windeployqt.exe` 部署依赖。

## 操作

- 左侧工具栏可导入显微图片、旋转图片、导入 OBJ/PLY 模型。
- 在图像区域单击可添加选区；使用加选、减选、撤销和清除完成编辑。
- 单击“生成3D模型”，确认积分后会进入生成流程并显示可操作的三维结果。
- 在 3D 区域按住左键旋转，按住右键或中键平移，滚轮缩放，双击复位。
- `O` 导入图片，`M` 导入模型，`Ctrl+S` 保存 PLY，`F11` 切换全屏，`Esc` 退出弹窗或全屏。

OBJ、ASCII PLY 与二进制小端 PLY 均可直接导入。示例流程使用设计稿素材；导入自己的图像后，界面继续使用同一套动态控件和选区逻辑。

## 视觉校验参数

可通过命令行直接打开各设计状态，并使用内置截图参数生成无窗口边框的 1280 × 800 对照图：

```bat
SAM3DQtClient.exe --state=result --screenshot=result.png
```

支持的状态为 `waiting`、`selected`、`confirm`、`generating`、`failed`、`result`、`saved`。
