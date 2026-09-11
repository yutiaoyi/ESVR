# Vorts 90 帧：完整动态训练记录

完成日期：2026-09-10（本机时间）。分支：`codex/deformable-esvr`。

## 结论

**已完整运行 40,000 个优化步骤，并完成全部 90 帧的全分辨率评估与导出。**
这是当前固定椭球数量、仅几何形变模型的第一轮完整真实数据训练。
平均逐帧 PSNR 为 **22.352944 dB**，切片可见明显细节丢失和噪声，
**尚不能认为实现了高质量动态重建，也不代表已经完全收敛**。

全部 90 帧都参与训练；下面的指标是训练帧重建指标，不是留出时间点的
泛化能力，也不能作为 Deformable 3DGS 原论文 RGB 基准结果。

## 数据检查

- 目录：`data/xueliao_timevaryingdata/vorts_timevarying`。
- 文件：`vorts0001.raw` 至 `vorts0090.raw`，90 个均存在。
- 每个文件 8,388,608 字节，总计 754,974,720 字节（720 MiB）。
- 每帧 128×128×128，little-endian float32，存储顺序 ZYX / X 最快。
- 所有值有限，90 帧共同范围 [-1,1]，统一映射到 [0,1] 训练，没有额外逐帧归一化。
- 第一帧与之前的 `data/xueliao_data/vorts0001.raw` SHA256 一致。
- 假定编号 1–90 为等间隔时间，网络时间归一化为 `(编号-1)/89`。
  尚无物理时间间隔信息，因此这些编号不代表秒。
- 每帧 SHA256 记录在运行目录的 `data_sha256.json`。

## 训练配置

运行目录：`output/dynamic/vorts90_full_geometry`。
复现脚本：`experiments/run_vorts_dynamic.sh`。
数据清单：`experiments/vorts_90_all_train.json`。

| 项目 | 本次配置 |
|---|---|
| 初始化 | 之前第一帧静态 ESVR 的 iteration_22400 PLY |
| 初始化文件 | `output/experiments/vorts0001_baseline/vorts0001_chunk_0/point_cloud/iteration_22400/point_cloud.ply` |
| 椭球数量 | 21,401，固定数量；无动态增密/剪枝 |
| 形变网络 | 8 层，256 宽；XYZ/time 各 10 个 Fourier 频带 |
| 形变输出 | 位移、旋转、对数尺度；不启用时间变化的标量分支 |
| 标量参数 | 规范空间标量仍参与优化，但跨时间共享 |
| 训练步骤 | 40,000；从已有 PLY 启动，warmup=0 |
| 空间采样 | 64³ 非重叠网格块随机采样，包含边界 |
| 时间采样 | 90 帧随机均匀采样；seed=42 |
| Canonical LR | 5e-5，指数衰减至 5e-6 |
| Deformation LR | 1.6e-4，指数衰减至 1.6e-5 |
| 损失 | 0.8 L1 + 0.2 (1-已有 ESVR SSIM) + 1e-4 形变平方惩罚 |
| 时间噪声 | 关闭 |
| 检查点 | 每 500 步保存 latest；每 10,000 步保留独立快照 |
| 监测 | 每 2,000 步评估第 1、46、90 帧全体积 |
| 设备 | RTX 5090；Python 3.10；Torch 2.7.1+cu128 |

训练日志耗时 431.691 秒（约 7.19 分钟，包含阶段监测，不包含最终导出）。
每帧实际采样 395–503 次，累计恰好 40,000 次；不存在完全未训练的帧。
这些步骤是优化步骤，**不是**静态 `train_chunk.py` 的扫描轮数。

## 最终全体积指标

所有指标使用未裁剪的预测值，归一化数据范围为 1。

| 指标 | 数值 |
|---|---:|
| 逐帧 PSNR 算术平均 | 22.352944 dB |
| 逐帧 PSNR 中位数 | 22.396394 dB |
| 最低 / 最高逐帧 PSNR | 19.758816 / 23.748214 dB |
| 合并所有体素 MSE 后的 PSNR | 22.311370 dB |
| 平均归一化 MAE | 0.05705103 |
| 原始 RAW 数值单位的合并 RMSE | 0.15327152 |
| 第 1 帧 PSNR | 23.234364 dB |
| 第 46 帧 PSNR | 22.137878 dB |
| 第 90 帧 PSNR | 19.758816 dB |

与刚启动时相比，中间和末尾时间步明显改善，但第一帧从专用静态模型
转为跨时间共享表示后质量大幅下降。此现象及切片误差说明当前方案存在
明显拟合不足/跨时间折衷；并不能仅凭这一轮实验确定是容量、参数化还是
优化策略的主因。若继续改进，应以这些已保存结果为基线进行受控比较。

## 输出位置

以下路径均相对于 `output/dynamic/vorts90_full_geometry/`：

- `checkpoint.pt`：完整模型、优化器与随机状态，8,954,429 字节。
- `checkpoint_010000.pt`、`checkpoint_020000.pt`、`checkpoint_030000.pt`、
  `checkpoint_040000.pt`：阶段快照。
- `model_inference.pt`：去除优化器的推理模型，2,987,423 字节。
  这只是 float32 序列化大小，不是已实现熵编码的压缩结果。
- `config.json`、`manifest.json`、`provenance.json`：参数、数据清单、代码 SHA256 与环境版本。
- `train.jsonl`、`monitor.jsonl`、`frame_sampling_counts.json`：训练过程与覆盖统计。
- `summary.json`、`evaluation/metrics.json`：汇总与全部 90 帧的详细指标。
- `evaluation/frame_0000.npy` 至 `frame_0089.npy`：归一化 XYZ 体数据，未裁剪。
- 同名 `.raw`：恢复 [-1,1] 对应的 RAW 数值单位，little-endian float32，ZYX / X 最快，未裁剪。
- 同名 `.ply`：各时间步形变后的椭球快照。
- 270 张 `*_axis{0,1,2}_gt_pred.png`：三个方向的离线 DVR，GT 左、预测右。
- `training_frames_dvr.gif`：全部 90 帧 DVR 对比，10 fps 仅为播放速度，不代表物理时间。
- `slice_comparison.png`：第 1、19、37、54、72、90 帧的中央 Z 切片与绝对误差。
- `quality_curves.png`：逐帧 PSNR 与训练阶段质量曲线。

注意导出文件的零起始下标：`frame_0000` 对应用户的 `vorts0001`，
`frame_0089` 对应 `vorts0090`；确切对应关系也记录在 metrics.json 中。
PNG/GIF 使用固定显示映射并裁剪到 [0,1]，不会用于计算指标。

## 验证

- 训练进程正常退出，40,000 步完成；90 帧指标和全部 RAW/NPY/PLY/PNG 均存在。
- `training_frames_dvr.gif` 实际含 90 帧。
- 精简模型独立重载并推理 time=46，与 `evaluation/frame_0045.npy` 的
  最大绝对差为 **0.0**。
- 第 46 帧 RAW 导出与 NPY 恢复原始数值范围后的最大绝对差为 **0.0**。
- 小规模断点续训回归测试与未中断重建的最大绝对差为 5.96e-8
  （CUDA 浮点累加误差量级）。
- 原有 CUDA 单元测试全部通过，Python 编译和 `git diff --check` 通过。
- 已人工查看质量曲线和中央切片：图像反映明显残留误差，没有将其标为高质量结果。

推理示例（无需加载原始 90 帧数据）：

```bash
.venv/bin/python inference_dynamic.py \
  --checkpoint output/dynamic/vorts90_full_geometry/model_inference.pt \
  --time 46 --output output/dynamic/my_frame46
```

训练和汇总复现（目标输出目录必须为空；现有结果应保留）：

```bash
bash experiments/run_vorts_dynamic.sh
.venv/bin/python experiments/summarize_dynamic_run.py \
  --run output/dynamic/vorts90_full_geometry \
  --manifest experiments/vorts_90_all_train.json
```

此脚本中的固定输出目录已含本次结果，重新训练前应改成一个新目录，
不能直接覆盖本次实验。

## 2026-09-11 补充：3D 与 2D 指标，以及论文口径

此前的 22.352944 dB 是 **3D PSNR**：每帧对 128³ 所有体素计算 MSE，
PSNR = -10 log10(MSE)，归一化 data_range=1，然后对 90 帧 PSNR 取算术平均。
不是切片 PSNR，不是渲染图 PSNR。先合并所有时间步 MSE 再取对数则为
22.311370 dB，两种聚合方式不可混称。

ESVR 论文主表/补充实验报告 3D PSNR、模型大小、2D PSNR/SSIM/LPIPS、
渲染 FPS、训练时间；部分补充实验还报告解码时间、图元数量、内存。
论文图像评估采用 800×800 渲染，FPS 以 0.5°步进的 720 视角旋转测量。
来源：[ESVR 原文，§5 与附录 D](https://arxiv.org/html/2608.05564)。

本次使用保存的重建体数据补算，没有重新训练、没有改动训练权重。

| 指标 | 实测值 | 口径 |
|---|---:|---|
| 3D PSNR ↑ | 22.352944 dB | 90 帧算术平均；完整体数据、未裁剪 |
| 3D SSIM ↑ | 0.609805 | 90 帧平均，额外补充的体结构指标 |
| 2D PSNR ↑ | 31.164083 dB | 270 张离线 DVR 图像平均 |
| 2D SSIM ↑ | 0.924334 | 同上 |
| LPIPS-Alex ↓ | 0.145573 | 同上；LPIPS v0.1，预训练 AlexNet |
| 归一化 MAE ↓ | 0.05705103 | 之前已记录，体数据 |
| 原 RAW 单位 RMSE ↓ | 0.15327152 | 之前已记录，合并全部体素 |
| 推理模型大小 | 2.987423 MB | 十进制 MB；包含 canonical 与 MLP，不含优化器 |
| RAW / 模型存储比 | 252.7177 : 1 | 754,974,720 / 2,987,423 字节；有损表示，不是熵编码压缩结果 |
| 图元数 | 21,401 | 所有时间共享的 canonical 图元数 |
| 训练时间 | 431.691 秒 | 已有 PLY 初始化后的动态训练，含阶段监测，不含第一帧预训练 |
| 完整体数据解码 | 平均 14.8677 ms / 帧 | RTX 5090，驻留模型，8 个 64³ 查询并在 GPU 组装 |
| 论文口径渲染 FPS | 未测 | 当前不是论文的 per-primitive ray sampling 渲染评测 |

### 必须保留的限定

- 2D 指标：90 时间步 × XYZ 三个正交视角，共 270 张 128×128 RGB 图。
  从 float32 体数据重新渲染，未经过 PNG 量化；使用既有 `dvr` 的固定 TF：
  alpha=1-exp(-0.18*clip(v,0,1))，颜色映射与之前预览相同。
  不是原论文 800×800 图像/相机设置，不能与原论文表格直接比较。
- 2D SSIM：skimage，11×11 Gaussian，sigma=1.5，population covariance，RGB 通道平均。
- 3D SSIM：现有 `query.compute_ssim_3d`，7³ Gaussian，sigma=1.5，零填充，
  population covariance，完整未裁剪体数据；是新增报告项，不冒充原论文主表 SSIM 定义。
- LPIPS：`lpips==0.1.4`，指标版本 0.1，AlexNet 预训练权重，输入 RGB 映射到 [-1,1]。
  [LPIPS 官方说明](https://github.com/richzhang/PerceptualSimilarity)。
  Deformable-3D-Gaussians 的本地 `paper` 分支评测代码使用 VGG，故本次 AlexNet
  数字也不能冒充其同配置 LPIPS；骨干名称必须随结果一并报告。
- 解码时间：预热 3 次，再测 90 个时间步；每次同步 CUDA，以 wall time 计时。
  不含模型加载、磁盘写入、GPU→CPU 拷贝或 DVR；不能取其倒数冒称端到端渲染 FPS。
- 图像遮挡与 TF 会掩盖内部误差，2D PSNR/SSIM 较高不推翻 3D 拟合不足的结论。
- 全部数据均为训练时间步，仍没有时间留出泛化测试。

完整逐体、逐图和计时原始值位于：
`output/dynamic/vorts90_full_geometry/supplemental_metrics/{summary,volumes,images,decode_ms}.json`。

补算命令：

```bash
.venv/bin/python experiments/evaluate_paper_metrics.py \
  --run output/dynamic/vorts90_full_geometry \
  --manifest experiments/vorts_90_all_train.json
```

新增评测依赖为 torchvision 0.22.1+cu128、lpips 0.1.4；原有 Torch 2.7.1+cu128
未更换。权重缓存位于被忽略的 `.tools/torch-cache`。
