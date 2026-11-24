# `models/` — 本地 ONNX 权重（**不入库**）

这里放 `demo.py` 直接加载的 `.onnx` 图。目录内容被 `.gitignore` 忽略（`models/*`，只有本文件入库），
所以换机器时不需要 clone 这些大文件，按下面重新下载即可。

## 放什么、从哪来

全部来自 Hugging Face 仓库 [`momolesang/InstantHMR`](https://huggingface.co/momolesang/InstantHMR)：

```
https://huggingface.co/momolesang/InstantHMR/resolve/main/<文件名>
```

最快的做法是让脚本帮你下（会落到 HF 缓存，不写这个目录）：

```bat
setup_windows.bat                                       :: medium-commercial + fit2 + mhr-body
setup_windows.bat --model medium --fit 0,1,2
.\.venv\Scripts\python.exe tools\prefetch_models.py --model nano --fit 2
```

想放到本目录（离线/多机拷贝）就手动下载后改名如下：

| 文件 | 用途 | 大小 | 许可 |
| --- | --- | --- | --- |
| `instanthmr-medium-commercial.onnx` | 网络本身（`--fit 0`） | 123.4 MB | 可商用 |
| `instanthmr-medium-commercial-fit1.onnx` | `--fit 1` | 79.3 MB | 可商用 |
| `instanthmr-medium-commercial-fit2.onnx` | `--fit 2`，根关节最稳（默认） | 79.4 MB | 可商用 |
| `mhr-body.onnx` | MHR body 图，解码 18,439 顶点 mesh | 12.7 MB | 随 LICENSE-SAM |
| `instanthmr-medium.onnx` / `-fit1` / `-fit2` | 精度更高，**非商用** | 93.7 / 79.3 / 79.4 MB | SAM License |
| `instanthmr-{nano,tiny,small}*.onnx` | 更小更快 | 56 ~ 65 MB | 非商用 |

## 命名约定（自己导出时）

`tools/export_onnx.py` 产出的名字建议照这个格式，`demo.py` 的 `--fit` 才和文件名对得上：

```
instanthmr-<档位>.onnx          # 网络本身
instanthmr-<档位>-fit1.onnx     # --fit 1
instanthmr-<档位>-fit2.onnx     # --fit 2
```

用的时候要么给名字让 demo 去 HF 缓存找（`--model medium-commercial --fit 2`），
要么给路径（`--model models\instanthmr-medium-commercial-fit2.onnx --fit 2`）。**`-fitN` 必须配 `--fit N`。**

## 不要放

- `.rrd` 回放、截图、视频：那是产物，放 `assets\` 或别处（`.rrd` 会长到 GB 级）。
- 训练权重 `.pth` / `.pt` / `ckpt`：放 `pytorch\`（见 `pytorch/README.md`）。
