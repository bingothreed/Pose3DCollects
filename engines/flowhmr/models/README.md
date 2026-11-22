# FlowHMR 模型文件

> Pose3D Collects 默认使用 `engines/flowhmr/models/`；可在 **设置与资源 → FlowHMR → 模型目录** 修改。本文件是占位说明，模型文件本身被 gitignore。

打开 FlowHMR 右上角的 **模型**，点 **下载全部公开模型**，或点某一行的 **下载**。程序会校验文件并放到对应目录，停止后可继续。**HaMeR** 包含在全部公开模型中；只做基础动捕时，可按行下载其他公开组。需要登录的官方包请手动下载。桌面推理开始生成时才加载模型。使用某一项高级功能时，可在「模型」里单独下载该项的权重。SAM 2.1、躯干 base 和 GeoCalib 的下载入口都在各自一行。

登录 [MANO 官网下载页](https://mano.is.tue.mpg.de/download.php)，按下表下载官方压缩包。打开 **模型 → 导入模型压缩包…**，直接选中下载的文件。程序会内置解压、识别模型组、复制到正确位置并列出结果，原包保留。若已自行解压，也可用 **导入已解压文件夹…**，或按表手动放置。

| 用途 | 官网条目和压缩包 | 解压到 | 解压后的位置 |
| --- | --- | --- | --- |
| **男／女体型，亦可用于手指精修** | **Models & Code**（模型和代码），`mano_v1_2.zip` | `models/` | `models/mano_v1_2/models/` 内的 `.pkl` |
| **可选中性体型** | **Extended SMPL+H model**（用于 AMASS 数据集），`smplh.tar.xz` | `models/body_models/smplh/` | `models/body_models/smplh/neutral/model.npz` |

基础动捕安装一个官方体型即可：有 `mano_v1_2.zip` 里的男或女体型，就不必再下载中性包。两个包都装好时，新任务优先使用中性；已有任务保留当时使用的参考体型。手动解压如果只得到 `smplh.tar`，可直接在软件中导入该 TAR。`models/body_models/smpl_neutral_J_regressor.pt` 是单独的公开文件，程序可一键下载。

```text
models/
├── flowhmr_latest/
│   ├── flowhmr_latest.ckpt
│   └── config.yml
├── flowhmr_base/                 ← 可选的躯干动作优化检查点
│   ├── flowhmr_base.ckpt
│   └── config.yml
├── yolox/
│   └── yolox_l.pth
├── sam-3d-body-dinov3/
│   ├── model.ckpt
│   ├── model_config.yaml
│   └── assets/mhr_model.pt
├── vggt-omega/
│   └── vggt_omega_1b_512.pt
├── geocalib/
│   └── pinhole.tar           ← 原片相机的图像重力模型
├── sam2/
│   └── sam2.1_hiera_small.pt ← 人物交叉优化模型
├── body_models/
│   ├── smpl_neutral_J_regressor.pt
│   └── smplh/neutral/model.npz   ← 可选的中性体型
├── mano_v1_2/models/            ← 官方男／女体型和 MANO 手部模型
└── hamer/
    ├── hamer.ckpt
    ├── model_config.yaml
    └── mano_mean_params.npz
```

`download-models.cmd` 下载公共文件。其中 [SAM 3D Body DINOv3](https://www.modelscope.cn/models/facebook/sam-3d-body-dinov3) 和 [VGGT-Omega](https://www.modelscope.cn/models/facebook/VGGT-Omega) 从魔搭下载并校验 SHA-256，无需 Hugging Face 授权。固定机位生成动作时跳过 VGGT 推理。使用原片相机反求时，部分固定机位任务的运镜镜头也会使用 VGGT。

### 躯干动作优化（可选模型）

生成设置里的 **躯干动作优化** 需要可选的 **FlowHMR 躯干模型（base）**：在 **模型** 面板找到同名一行，点 **下载**（约 1.72 GiB，来自与主模型相同的官方 Hugging Face 仓库并核对 SHA-256）。安装后开关变为可用，未安装时开关会禁用并提示。

躯干动作优化默认关闭。开启后，程序会额外用 base 检查点生成一次，取骨盆、脊柱、颈部和头部的局部旋转，并以标准结果的脚部世界位置和朝向为目标调整双腿。手指和其他关节仍来自当前的主模型；需要可选模型时也可把 `flowhmr_base` 文件夹放入 `models/` 后点 **刷新**。

### 人物交叉优化（实验性）

在「模型」面板找到 **SAM 2.1 人物分割（交叉优化）** 并下载。程序会安装所需的权重和推理组件。就绪后可在生成设置中开启「人物交叉优化」。它用于辅助处理交叉和遮挡时的人物追踪。

### 原片相机反求（实验性）

在「模型」面板下载 **GeoCalib · 图像重力** 后，可以在任务的「相机反求」栏重算机位。缺少 GeoCalib 时可使用人物与地面估计。相关提示在原片机位预览和相机报告中显示。[GeoCalib 模型说明](../docs/geocalib.md)

### HaMeR 手指精修

1. 在 **运行环境** 启用可用的 GPU 推理环境。
2. 登录 [MANO 官网](https://mano.is.tue.mpg.de/download.php)，下载 **Models & Code**（`mano_v1_2.zip`）。在 **模型 → 导入模型压缩包…** 直接选择 ZIP。精修读取原版 `models/mano_v1_2/models/MANO_RIGHT.pkl`；可选中性包 `smplh.tar.xz` 不包含它。
3. 在 **模型** 面板点击 **HaMeR** 一行的下载按钮。程序优先从[镜像](https://huggingface.co/Leo-TX/hamer/tree/f64df318d49e7e014a6a7f5d0547cba87d6e4317)下载所需的三个文件（约 2.69 GB），逐个核对与作者包相同的 SHA-256。若镜像不可用，会尝试[作者的 `hamer_demo_data.tar.gz`](https://www.cs.utexas.edu/~pavlakos/hamer/data/hamer_demo_data.tar.gz)（约 6.04 GB）。已有作者压缩包时也可直接导入。
4. HaMeR 显示 **已就绪** 后，在生成设置中开启 **HaMeR 手指精修**。如果提示 MANO 文件错误或不完整，请用官方包内的 `MANO_RIGHT.pkl` 原文件替换后点 **刷新**。
