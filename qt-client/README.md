# SAM 3D Qt 客户端

这是一个独立的 Qt Widgets + OpenGL 桌面客户端，按 `design` 中的 1280 × 800 设计稿实现真实的函数计算工作流：点选调用 `POST /segment`，使用函数返回的 Mask 调用 `POST /generate`，最后解析、显示并保存 GLB 2.0 模型。

## 构建

项目使用 Qt 5.15.2 `msvc2019_x64`。在 Visual Studio x64 开发环境中执行：

也可以直接双击 `run-release.bat`；脚本会先执行一次增量构建再启动，避免运行到旧的 EXE。仅需构建时可单独双击 `build-release.bat`。

```bat
mkdir build
cd build
D:\Qt\5.15.2\msvc2019_x64\bin\qmake.exe ..\SAM3DQtClient.pro CONFIG+=release
nmake
```

运行时请将 `D:\Qt\5.15.2\msvc2019_x64\bin` 加入 `PATH`，或使用同目录下的 `windeployqt.exe` 部署依赖。

## 操作

- 点击右上角“选择图片”、按 `Ctrl+O`，或将一张本地图片拖入窗口即可换图；左下角图片按钮也可打开文件选择器。支持 PNG、JPEG、BMP、TIFF、WebP（以本机 Qt 图像插件为准），自动应用照片方向。
- 自选图片完整适配画布，空白边缘不接受点选。换图会清空旧选区和模型、取消旧请求并退出截图预览状态，新图片可继续捕捉和生成。
- 左侧工具栏可旋转图片和配置 SAM3/SAM3D 组合函数地址。
- 在图像区域左键增加选区、右键减少选区，也可先切换“增加选区”或“减少选区”再单击；每次编辑都会携带完整正负点列表调用 `/segment`。
- 只有函数返回了与原图同尺寸的 PNG Mask 后才能生成；确认后程序调用 `/generate` 并校验 GLB 2.0 文件头。
- 在 3D 区域按住左键旋转，按住右键或中键平移，滚轮缩放，双击复位。
- `Ctrl+O` 选择图片，`M` 导入本地 GLB，`Ctrl+S` 保存 GLB，`F11` 切换全屏，`Esc` 退出弹窗或全屏。

启动时也可直接选择图片：`SAM3DQtClient.exe --image="D:\images\显微图片.jpg"`。图片读取失败会保留当前画布；启动参数中的图片无效时程序以退出码 2 结束。

默认函数地址与网页客户端一致。可以通过界面设置、环境变量 `SAM3D_SERVICE_URL` 或启动参数 `--endpoint=https://...` 覆盖；界面设置会持久保存。运行时只接收和导出 GLB 2.0。

## 视觉校验参数

视觉回归时可显式打开只用于截图的预览状态，并生成无窗口边框的 1280 × 800 对照图。正常启动不会进入预览状态：

```bat
SAM3DQtClient.exe --state=result --screenshot=result.png
```

支持的状态为 `waiting`、`selected`、`confirm`、`generating`、`failed`、`result`、`saved`。

界面按设计图统一中文字体、悬浮工具栏、选区描边、蓝色渐变按钮和各状态弹窗；确认框展示实际转换信息，尚未接入积分计费。截图按窗口的逻辑像素输出，在 Windows 150% 等缩放下仍为 1280 × 800。

`--state` 是只用于视觉校验的预览：选区来自设计参考，3D 区域使用程序生成的示例网格，不调用分割或生成服务。点击“选择图片”或使用 `--image` 后退出预览，恢复真实的图片、Mask 和 GLB 工作流。真实生成期间进度条不显示虚构百分比。
