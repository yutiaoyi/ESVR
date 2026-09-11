# 官方附带 TF 表：独立 2D 重评估

## 结果与适用范围

对同一 40,000 步模型、同一 90 帧重建体数据，使用官方仓库附带
`SIBR_viewer/install/bin/transferFunc.txt` 重新渲染并评估。
没有重新训练，没有删除或覆盖旧评估。

**这次是“官方附带 TF 表的离线 DVR 适配”，不是论文完整渲染协议复现。**
尚未确认该表对应论文哪个数据集/实验，当前仓库未找到用于核查 LUT
索引、透明度修正等细节的定制 CUDA 内核。保持此前三视角和分辨率，
用于比较更换 TF 后的指标；不冒称论文的 800×800 / 720 视角结果。

| TF 设置 | 平均 2D PSNR ↑ | 平均 2D SSIM ↑ | 平均 LPIPS-Alex ↓ |
|---|---:|---:|---:|
| 旧：自定义预览 TF | 31.164083 dB | 0.924334 | 0.145573 |
| 新：官方附带 LUT，离线适配 | 15.750268 dB | 0.250386 | 0.280273 |

均为 90 个训练时间步 × XYZ 三个正交视角，共 270 对 128×128 RGB 图像。
先逐图计算，再对指标取算术平均；浮点图像评测在 PNG 量化之前完成。
SSIM 使用 Gaussian sigma=1.5 / 11×11 / population covariance / RGB 通道平均；
LPIPS 使用 v0.1 预训练 AlexNet。未更换模型，3D PSNR 仍为 22.352944 dB。

新 TF 下图像误差明显增大；已查看第一帧 Z 方向对比，细结构及色彩分布
差异明显。这支持旧单一 TF 的高图像分数不能代表普遍渲染质量的结论，
但不能据此宣称新 TF 是 vorts 数据唯一正确或最合适的 TF。

## 官方设置的证据与实际执行规则

- 论文 §4.4 描述射线标量通过 TF 映射为颜色/透明度再合成；附录 B.2
  给出交互 TF 编辑器：[ESVR 原文](https://arxiv.org/html/2608.05564)。
- `GaussianView.cpp` 实际 include 的是 `tfeditorGUI.h`，而不是同目录
  的 `tfeditorGUI_1D.h`。前者 `init_transferFunction()` 在工作目录存在
  `transferFunc.txt` 时调用 `loadTransferFunctionFromFile()`。
- 该函数读取 RGBA 四列，将第 i 行设为标量位置 `i/255.0`，最多 256 行。
  编辑器按控制点生成线性插值 RGBA 表，GPU 接口接收该表。
- 本次固定使用 install/bin 附带的全部 256 行，不改色、不调透明度曲线。
  输入强度从原始 [-1,1] 统一归一化到 [0,1]，渲染时裁剪到此范围；
  离线查询按 `255*s` 线性插值 RGBA。
- 官方原表最后一行 alpha=1.005144。本次保留源文件原值和快照，但
  在合成前将插值后的 alpha 裁剪到 [0,1]；这是明确的安全处理，
  未核实原生 CUDA 对该异常值的处理，不宣称完全等价。
- 一体素一步、无光照、黑色背景、前向透明度累积；直接使用表内 alpha，
  **不再套用旧的 1-exp(-0.18*s)**，没有额外采样步长透明度修正。

TF 文件 SHA256：
`d2e0ea8a0f84c6ece8687891b156b42aeea9595a946db14c49a10ee87359a60a`

## 旧结果保留与标注

旧 `supplemental_metrics/` 的指标文件未改动；summary.json 重评估前后
SHA256 均为 `63f65ffed64883aebe0b303909084e04a597b2eb262f35f15013daf1af6e607a`。
新增 `supplemental_metrics/TF_PROTOCOL.md` 注明旧结果使用：

```text
s = clip(normalized_scalar, 0, 1)
RGB(s) = (0.95, 0.22 + 0.7*s, 0.08 + 0.7*s)
alpha(s) = 1 - exp(-0.18*s)
```

原 `evaluation/` 内的渲染 PNG 及根目录的原 GIF 也采用这个自定义 TF，
均保留，不应与本次新输出混用。

## 新输出与复现

目录：`output/dynamic/vorts90_full_geometry/official_tf_metrics/`

- `summary.json`：新旧指标与全部协议说明。
- `images.json`：270 张图片的逐图指标。
- `transferFunc.txt`：原官方 TF 的逐字节快照。
- 270 张 `frame_0001_axis0_gt_pred.png` 等对比图：GT 左、预测右。
  此处编号 0001–0090 与原始 vorts 文件一致。

```bash
.venv/bin/python experiments/evaluate_official_tf.py \
  --run output/dynamic/vorts90_full_geometry \
  --manifest experiments/vorts_90_all_train.json \
  --output output/dynamic/vorts90_full_geometry/official_tf_metrics
```

脚本拒绝写入非空目标目录；再次运行应选一个新目录。执行前已通过 LUT
端点/中点插值、常量透明度体前向合成的解析检查；评估正常退出，270 图完成。
