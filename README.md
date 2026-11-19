# EISeg

[![Python 3.6](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/release/python-360/) [![PaddlePaddle 2.2](https://img.shields.io/badge/paddlepaddle-2.2-blue.svg)](https://www.python.org/downloads/release/python-360/) [![License](https://img.shields.io/badge/license-Apache%202-blue.svg)](LICENSE) [![Downloads](https://pepy.tech/badge/eiseg)](https://pepy.tech/project/eiseg)
<!-- [![GitHub release](https://img.shields.io/github/release/Naereen/StrapDown.js.svg)](https://github.com/PaddleCV-SIG/iseg/releases) -->

简体中文 | [English](README_EN.md)

## 本仓库新增的功能

这个仓库是在 EISeg 1.1.x 基础上 fork 的，面向工业缺陷标注场景，新增了两个功能：

### 1. 显示当前图片序号

状态栏右侧会常驻显示 **`第 i / N 张 | 已标注 M 张`**：

- `i / N`：当前标注的是文件夹里的第几张图，共几张；
- `已标注 M 张`：标签目录里已经有结果的图片数量，方便判断还剩多少没标；
- 窗口标题同步显示 `文件名 (i/N)`；
- 翻页、点选列表、打开文件夹、保存标签后都会自动刷新。

### 2. 导入 SAM 模型（Segment Anything）

「模型选择」面板新增 **导入SAM模型** 按钮（菜单「文件 → 导入SAM模型」，快捷键 `Ctrl+Shift+D`），
可以直接加载 Meta 官方 SAM 权重 `sam_vit_b / sam_vit_l / sam_vit_h`（`.pth`），
用左键正点、右键负点的方式做交互式分割，替代需要单独下载的 PaddleSeg 静态图模型。

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install --no-deps git+https://github.com/facebookresearch/segment-anything.git
```

权重下载地址与完整用法见 [docs/sam.md](docs/sam.md)。SAM 是可选后端，
不安装 `torch` / `segment-anything` 时 EISeg 仍可正常使用原有的 PaddleSeg 模型。

## 最新动向

- [2022.7.20] 为了减少重复工作以集中精力做好后续的开发和维护工作，**后续EISeg将仅在飞桨官方仓库[PaddlePaddle/PaddeSeg](https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.6/EISeg)下进行更新，现有代码仓库将不再更新**。非常感谢大家的一路相伴与支持，EISeg将竭诚为大家提供更多更好用的标注功能，欢迎大家去新的地址体验，也希望大家继续支持。
