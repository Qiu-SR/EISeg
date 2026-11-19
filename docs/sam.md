# SAM 交互式分割后端

在原有 PaddleSeg 静态图模型之外，本仓库额外支持把 Meta 的
[SAM (Segment Anything)](https://github.com/facebookresearch/segment-anything)
当作交互式分割模型使用：导入一个 `.pth` 权重后，左键正点、右键负点即可直接出掩膜，
标注流程（完成目标、翻页、保存灰度图/伪彩色/JSON/COCO）与原来完全一致。

## 1. 安装依赖

SAM 需要 `torch` 和 `segment-anything`，两者都是**可选依赖**，EISeg 启动时不会导入，
没装也能正常使用 PaddleSeg 模型。

```bash
# CUDA 版 PyTorch（推荐，按显卡驱动选择 cu118 / cu121 / cu128）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 纯 CPU 版（下载小，但 SAM 出图很慢）
pip install torch torchvision

# segment-anything 只需要源码，不拉额外依赖
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
```

> `torchvision` 是必需的：`segment_anything` 的 `ResizeLongestSide` 用的是
> `torchvision.transforms.functional`，缺了会在导入权重时报 `No module named 'torchvision'`。

## 2. 下载权重

官方权重（任意一个即可，越大越准、越慢）：

| 权重 | 大小 | 下载地址 |
| ---- | ---- | -------- |
| sam_vit_b | 375 MB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth |
| sam_vit_l | 1.25 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth |
| sam_vit_h | 2.4 GB | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |

> 权重文件请放在**纯英文路径**下（与 PaddleSeg 模型的要求一致），
> 路径里带中文会在导入时被拦截并提示。

## 3. 使用

1. 启动 EISeg，在右侧「模型选择」面板点击 **导入SAM模型**（快捷键 `Ctrl+Shift+D`），
   或使用菜单「文件 → 导入SAM模型」。
2. 选择 `sam_vit_b_01ec64.pth` 之类的权重，等待状态栏提示「模型加载成功」。
   面板上的「当前模型」会显示 `SAM-vit_b（cuda）` 这样的后端与设备信息。
3. 打开图片文件夹，选中标签后：
   - **左键**：在目标内部添加正点；
   - **右键**：在目标外部/多余区域添加负点，用来缩小范围；
   - **空格**：完成当前目标，多边形落到画布上，接着标下一个；
   - `Ctrl+Z` / `Ctrl+Y`：撤销 / 重做一步点击。
4. 后续保存流程与 PaddleSeg 模型完全一样（保存灰度图 + 伪彩色 + 抠图 + COCO）。

## 4. 说明与注意事项

- SAM 的「使用掩膜」选项不生效（SAM 每次都由完整点击序列重新推理，不依赖上一轮预测），
  导入 SAM 后该复选框会自动置灰。
- 图像编码（image encoder）在每张图**第一次**被点击时执行一次，
  之后同一张图的每次点击只跑 mask decoder，GPU 上可以做到实时。
- 如果显存不够或驱动不匹配，可以强制走 CPU：

  ```bash
  set EISEG_SAM_DEVICE=cpu
  ```

- 环境变量 `EISEG_SAM_MULTIMASK=1` 可以让 SAM 在单点提示时返回 3 个候选掩膜并自动取
  分数最高的一个（默认关闭，默认使用 SAM 推荐的单一最优掩膜）。
- 支持直接加载官方 `sam_vit_b / sam_vit_l / sam_vit_h` 权重，结构会根据权重里的
  `image_encoder` 维度与层数自动识别，不需要手动选择模型类型。

### Windows 上与 PaddlePaddle 的 DLL 顺序问题

先 `import paddle` 再 `import torch` 会让 torch 的 `shm.dll` 加载失败
（`OSError: [WinError 127]`），顺序反过来则两者都正常。EISeg 在
`eiseg/__init__.py` 里会在导入 paddle 之前先尝试导入 torch，所以正常启动即可；
如果启动时设置过 `EISEG_NO_TORCH=1`，SAM 弹窗会提示这一点，去掉该变量重启即可。

