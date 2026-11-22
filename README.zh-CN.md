# Pose3D Collects

[English](README.md) | **简体中文**

一个在本机运行的人体动作重建工作台，把 **InstantHMR**、**FlowHMR** 和 **GEM-X** 集成到同一个界面里：

- **参考视频**：上传/预览视频，显示人物框与 2D 关键点。
- **3D 渲染**：用 Three.js 播放重建出的人体网格序列，可旋转、缩放、拖动时间轴、线框模式、下载 NPZ；还可开启「同步视频」让参考视频（含音频开关）以小窗跟随网格播放，或用「固定视角」把相机固定在场景中（默认关闭，相机会跟随人物移动，类似 FlowHMR 的「跟随」）。
- **设置与资源**：检测引擎环境、一键自动配置运行环境（FlowHMR 便携版会按显卡选择并下载 PyTorch）、自动/手动拉取模型、启动/停止 FlowHMR 服务。

两个引擎各自使用自己的 Python 环境，Hub 只做进程编排和 HTTP 代理，因此不会出现 torch / onnxruntime 版本冲突。

> 项目目录目前叫 `HMRCollects`，产品名是 **Pose3D Collects**。把整个文件夹重命名为 `Pose3DCollects` 不影响运行（所有路径都是相对或配置项）。

---

## 快速开始

双击 **`start.bat`**。

首次运行会自动创建 Hub 用的 `.venv` 并安装 Flask（只有这一个依赖），然后打开 <http://127.0.0.1:7860>。

也可以手动启动：

```bat
.venv\Scripts\python.exe app.py --open
.venv\Scripts\python.exe app.py --port 8080
```

需要 Python 3.10+。不想自动打开浏览器时，先 `set POSE3D_NO_OPEN=1` 再运行 `start.bat`。

### 独立窗口

双击 **`start-desktop.bat`**：

1. 优先用 pywebview + Edge WebView2 打开原生窗口（首次会安装可选依赖 `pywebview`）。
2. 如果 WebView2 不可用，自动退到 Edge/Chrome 的 `--app` 应用窗口（没有地址栏和标签页）。
3. 再失败才使用系统默认浏览器。

如果已经有 `start.bat` 的 Hub 在运行，桌面入口会直接复用；关闭桌面窗口不会停止那个 Hub。由桌面入口自己启动的 Hub 会随窗口关闭而停止。检查桌面组件：`start-desktop.bat --check`。

### 中英文切换

顶栏的 **EN / 中** 按钮在中文和英文之间切换，选择保存在浏览器本地（桌面窗口也会保留）。界面按钮、状态、任务卡片、设置表格和提示都会同步切换；引擎自身输出的日志和动态错误详情可能仍保持引擎原语言，可在「任务日志」中查看原文。

---

## 首次配置引擎

打开顶部 **设置与资源**，页面只显示当前顶栏选中的引擎；切换顶栏引擎即可查看对应的环境、模型和设置。模型目录默认指向仓库内置的引擎目录，且可以随时修改；每个模型目录都保留一份入库的 `README.md` 占位说明，列出应放置的文件。选择引擎/模型后，任务面板和设置卡片会列出缺失的环境项；环境不完善时显示 **自动配置环境** 按钮，完善后按钮变为 **环境已完善**。

### InstantHMR

InstantHMR 源码已内置到本仓库，运行器、环境和模型放在同一目录：

| 项目 | 默认值 |
|---|---|
| 项目目录 | `engines\instanthmr` |
| 虚拟环境 | `<项目>\.venv\Scripts\python.exe` |
| 模型目录 | `<项目>\models` |

1. 点 **自动配置环境**：自动创建 `.venv`，先从阿里云镜像安装 CUDA 12.6 版 PyTorch（无 NVIDIA 显卡时装 CPU 版），再执行 `pip install -r requirements.txt`，并校验运行依赖。需要 Python 3.12。
2. 在模型表里选默认模型和拟合方式（默认 `medium` + `fit2`），点 **下载**：从 Hugging Face 下载 `instanthmr-<size>-fit<fit>.onnx` 和 `mhr-body.onnx`。
3. 等状态变成 **已就绪**。

也可以手动把模型文件放进模型目录，然后点 **保存设置** 刷新检测。

在 **任务默认值** 里可以设置两层平滑：**姿态平滑**（默认 One-Euro，在生成网格前过滤根旋转和身体关节角）和 **输出平滑**（默认关闭，对关键点、网格和位移再做一层 1€ 滤波）。任务面板里也有 **输出平滑** 开关，可对单次任务覆盖默认值；开启后快速动作可能略有滞后。

**实时摄像头**：在 InstantHMR 设置卡里设置 **摄像头编号**（默认 0），点 **打开实时摄像头**。应用会用当前的模型、拟合方式、设备和平滑设置启动 InstantHMR 自带的 `demo.py --camera`：每采集一帧就推理一次，并在 Rerun 窗口里同步更新 2D/3D 结果——**不是先录制再重算**。检测器每 3 帧刷新一次，人体姿态仍逐帧推理；再点 **关闭实时摄像头**（或退出应用）结束。日志在 `data/logs/instanthmr-camera.log`。

### FlowHMR

FlowHMR-Windows 的**源码**已内置到 `engines\flowhmr`；运行时（Python 3.10 + PyTorch）在本目录创建，模型也下载到同一目录：

| 项目 | 默认值 |
|---|---|
| 项目目录 | `engines\flowhmr` |
| Python | `<项目>\.venv\Scripts\python.exe`（Python 3.10） |
| 端口 | `60550` |
| 模型目录 | `<项目>\models` |

- 设置页会用「模型根目录」标出放置位置，可复制路径或直接打开/创建文件夹；每个模型行尾也有「打开目录」。手动下载的模型按行内完整路径放入即可。修改模型目录后需要重启 FlowHMR 服务才会生效。
- 点 **自动配置环境** 会创建 Python 3.10 的 `.venv`，先从阿里云镜像安装 PyTorch/torchvision 2.8.0+cu128，再安装 `requirements-windows.txt`、Web/感知依赖和 YOLOX；不再需要便携版 `runtime\` 目录。
- 模型缺失时点 **下载公开模型**，会调用 FlowHMR 自带的下载脚本（含 SHA-256 校验）；完整模型约 20 GB。
- 官方需要登录/授权的体型包（MANO / SMPL-H）按 FlowHMR 的 `models/README.md` 手动下载，放到对应目录后点 **保存并检测**。

提交第一个 FlowHMR 任务时会自动启动服务；首次启动需要加载模型，可能要几分钟。服务日志在 `data/logs/flowhmr-server.log`。

### GEM-X

GEM-X（[NVIDIA](https://github.com/nv-tlabs/GEM-X)）按帧跑 **Body Pose**：YOLOX 检测并跟踪主要人物，DINOv3 + SOMA 模型在相机坐标系重建全身（18056 顶点 SOMA 网格、77 关节）。保留源视频帧率（不做 30Hz 归一化——那是 Body Motion 管线，暂未接入本工作台）。

| 项目 | 默认值 |
|---|---|
| 项目目录 | `engines\gemx` |
| Python | `<项目>\.venv\Scripts\python.exe` |
| 权重目录 | `<项目>\weights` |
| SOMA 资产 | `<项目>\assets\soma` |

- 点 **自动配置环境**：创建 `.venv` 并安装 GEM-X（`pip install -e ".[video]"`）。检测到 NVIDIA 显卡（`nvidia-smi`）时自动安装 CUDA 版 PyTorch（`torch 2.10.0+cu128`，走阿里云镜像，约 2.9 GB），否则用 CPU 版。本机需要 Python 3.12（固定使用的 CUDA 轮子是 cp312）。
- 点 **自动下载模型**（约 4.1 GB）：YOLOX HumanArt 检测器（OpenMMLab）、`body_s240000.ckpt` 与 `body_config.json`（Hugging Face `nvidia/GEM-X`）、SOMA 0.3.3 资产（Hugging Face `nvidia/SOMA-X` 的 `v0.3.3` revision，含校正权重）。**下载代理**字段用于 Hugging Face（默认 `http://127.0.0.1:21882`），YOLOX 压缩包直连下载。下载支持断点续传，已完成的文件自动跳过。
- 默认批量下推理约需 7 GB 显存；CPU 也能跑但很慢。首次任务会编译 SOMA 的 Warp 内核（约 10 秒），之后走缓存。环境安装会在 NVIDIA 机器上自动装 GPU 版 ONNXRuntime，检测器默认 **自动**（先 CUDA 失败回退 CPU——worker 会把 torch 自带的 CUDA DLL 加入 PATH，ORT-GPU 只缺这个）。
- **姿态批量**与 **fp16 加速**控制姿态吞吐（RTX 3060 实测：批量 16 + fp16 ≈ 55–60 ms/帧 ≈ 17 fps，顶点与 fp32 最大差 <0.3 mm）。CUDA 检测约 70–90 ms/帧，90 帧任务端到端约 30 秒（≈ 7 fps 流水线）——快，但不是实时；GEM-X 的 DINOv3-H 主干本身就不是实时模型。
- 试过并放弃的提速路径：TensorRT（ORT 的 TRT EP 无法构建 YOLOX 图——其 TopK K=8400 超过 TRT 上限 3840）、检测器 fp16 转换（3060 上无增益）、检测与姿态流水线并行（显存带宽竞争，两边都慢约 3 倍）。
- SOMA 跑 Windows 不在 GEM-X 上游验证范围内——本次接入在 RTX 3060（驱动 621.24，warp 1.19 / usd-core 26.11）上实测通过。

> GEM-X 目前每条视频只重建自动跟踪到的最高置信度一人。Body Motion（时序、世界坐标）、Hand Pose、多人框暂未接入工作台。

---

## 使用流程

1. 右上角选择引擎。
2. 右侧拖入视频，设置最大帧数（`0` = 全部；长视频耗时更久）和本次任务的精度：
   - InstantHMR：模型档位（Nano → Medium，越往后越准）+ 拟合方式（网络直出 / 1 crop / 2 crop）。
   - FlowHMR：生成精度（快速 10 步 / 标准 20 步 / 精细 40 步），步数越多动作越精细、耗时越长。
   - GEM-X：逐帧 Body Pose，无额外任务选项；设备与检测器在「设置与资源」里配置。
     还可按需开启模型增强（Windows 便携版，模型已下载时可选）：HaMeR 手指精修（需要 GPU）、躯干动作优化（`flowhmr_base`）、人物交叉优化（SAM 2.1）。这些会额外加载模型并增加耗时。
   所选模型未下载时，开始按钮会禁用并提示到「设置与资源」下载。
3. 点 **开始重建**。
4. **参考视频** 页显示进度；多人视频且开启了「多人时暂停让我选」时，会出现人物缩略图供选择。
5. 完成后点 **查看 3D 结果**，在 **3D 渲染** 页播放网格动画。

> InstantHMR 每条视频只重建置信度最高的一人（速度快、界面简单）；需要多人分别重建时请用 FlowHMR，它会列出人物缩略图供选择。
> FlowHMR 关闭「静态机位」时需要 VGGT-Omega 相机模型；模型缺失时程序会提示下载或改用静态机位。

最近任务保存在 `data/jobs/<job_id>/`，重启 Hub 后仍可查看。FlowHMR 的任务同时保存在它自己的 `output/web_jobs/` 下。

---

## 目录

```
HMRCollects/
├─ app.py                 Hub：Flask 路由、上传、状态、结果代理
├─ desktop.py             独立窗口外壳（WebView2 / 浏览器应用窗口）
├─ start.bat              一键环境 + 浏览器启动
├─ start-desktop.bat      一键环境 + 独立窗口启动
├─ requirements-desktop.txt  可选 pywebview 依赖
├─ engines/                内置引擎源码（模型/虚拟环境不入 git）
│  ├─ instanthmr/          InstantHMR 包 + demo.py
│  ├─ flowhmr/             FlowHMR 应用源码 + third_party
│  ├─ gemx/                GEM-X 包 + 打包文件
│  └─ tdpt/                TDPT VNect 运行器 + demo.py
├─ hub/
│  ├─ config.py           配置与内置引擎路径
│  ├─ engines.py          引擎就绪检查、环境/模型安装任务
│  ├─ flow_server.py      FlowHMR 服务生命周期 + HTTP 客户端
│  └─ jobs.py             统一任务模型、引擎 worker 调度
├─ workers/               统一结果写出器与模型下载脚本
├─ templates/index.html
├─ static/css/app.css
├─ static/js/             main / viewer / overlay / settings / api / state
├─ static/vendor/three/   本地 Three.js（MIT）
└─ data/                  配置、任务、日志、临时文件
```

### 统一结果格式

InstantHMR 的 worker 会把结果写成与 FlowHMR 相同的结构：

```
data/jobs/<id>/
├─ video.mp4         参考视频（有 FFmpeg 时转成浏览器可播放的 H.264）
├─ track.json        2D 人物框 / 关键点（按视频 fps）
├─ mesh/
│  ├─ meta.json      num_frames / fps / vertices / faces / bbox / root_trajectory
│  ├─ vertices.bin   float16, 帧优先
│  └─ faces.bin      uint32
└─ motion.npz        vertices / joints / joints_2d / boxes / faces / fps
```

FlowHMR 的结果通过 Hub 代理它的 `/api/jobs/<id>/meta|vertices|faces|track|video|npz`，前端不需要关心结果来自哪个引擎。

---

## 常见问题

- **视频能处理但浏览器不播放**：安装 FFmpeg 并在设置里填路径；Hub 会用 FFmpeg 转码成 H.264。FlowHMR 自带 FFmpeg（`imageio-ffmpeg`），通常会被自动找到。
- **InstantHMR 报 CUDA 不可用**：把设备改成 `CPU`，或确认 `onnxruntime-gpu` 和显卡驱动；实例化失败时 worker 会自动回退 CPU。
- **FlowHMR 服务启动失败**：看 `data/logs/flowhmr-server.log`，常见原因是端口占用、显存不足或模型文件不完整。也可以在设置里把设备改成 `cpu`。
- **下载很慢**：FlowHMR 模型可从魔搭/镜像下载，脚本会优先镜像；InstantHMR 模型在 Hugging Face 缓存和模型目录各有一份时可手动复制。
