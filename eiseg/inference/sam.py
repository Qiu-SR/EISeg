# Copyright (c) 2026 EISeg-SAM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Segment Anything (SAM) 交互式分割后端。

本模块把 Meta 的 SAM 包装成与 EISeg 原有 PaddleSeg 推理器一致的接口，
使得在「模型选择」里导入一个 ``.pth`` 权重后，直接左键正点、右键负点
进行交互式标注，而不再需要预先下载 PaddleSeg 静态图模型。

对外接口（由 ``InteractiveController`` 调用）：

- :meth:`SAMModel.build_predictor` -> :class:`SAMPredictor`
- :meth:`SAMPredictor.set_input_image`
- :meth:`SAMPredictor.get_prediction`
- :meth:`SAMPredictor.get_states` / :meth:`SAMPredictor.set_states`

依赖：``torch`` 与 ``segment-anything``，两者都是延迟导入，
没装时不会影响 EISeg 的启动，只有真正导入 SAM 权重时才报错。
"""

import os
import os.path as osp
import re

import numpy as np

# 各版本 SAM 的结构特征：(image_encoder 的嵌入维度, image_encoder.blocks 层数)
_ARCH_BY_SHAPE = {
    (768, 12): "vit_b",
    (1024, 24): "vit_l",
    (1280, 32): "vit_h",
}
_ARCH_BY_NAME = {
    "vit_b": "vit_b",
    "vit_l": "vit_l",
    "vit_h": "vit_h",
}
_SAM_WEIGHT_EXTS = (".pth", ".pt", ".ckpt", ".pth.tar")


def is_sam_weight_path(path):
    """判断一个路径看起来是不是 SAM 权重文件。"""
    if not isinstance(path, str):
        return False
    return path.lower().endswith(_SAM_WEIGHT_EXTS)


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError:
        raise ImportError(
            "导入 SAM 模型需要 PyTorch，当前环境未安装。\n"
            "请在 EISeg 所在的 Python 环境执行：\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
            "（无 NVIDIA 显卡时去掉 --index-url 参数安装 CPU 版）"
        )
    except OSError as e:
        raise ImportError(
            "PyTorch 加载失败（DLL 冲突）：%s\n"
            "Windows 上 PyTorch 必须先于 PaddlePaddle 导入，EISeg 启动时会自动处理。\n"
            "如果设置过 EISEG_NO_TORCH=1，请去掉该环境变量后重启 EISeg。" % e
        )


def _require_sam():
    try:
        import segment_anything  # noqa: F401
        from segment_anything import SamPredictor, sam_model_registry
    except ImportError:
        raise ImportError(
            "导入 SAM 模型需要 segment-anything 包，当前环境未安装。\n"
            "请在 EISeg 所在的 Python 环境执行：\n"
            "  pip install git+https://github.com/facebookresearch/segment-anything.git\n"
            "或：pip install segment-anything"
        )
    return SamPredictor, sam_model_registry


def _to_state_dict(obj):
    """兼容 {"model": state_dict} / {"state_dict": ...} 这类包装过的权重。"""
    if not isinstance(obj, dict):
        return obj
    for key in ("model", "state_dict", "model_state_dict"):
        inner = obj.get(key)
        if isinstance(inner, dict) and len(inner) > 0:
            return inner
    return obj


def detect_arch(state_dict, ckpt_path=""):
    """从权重里推断 SAM 结构（vit_b / vit_l / vit_h）。"""
    embed_dim, blocks = None, 0
    proj = state_dict.get("image_encoder.patch_embed.proj.weight")
    if proj is not None and hasattr(proj, "shape") and len(proj.shape) >= 1:
        embed_dim = int(proj.shape[0])
    for key in state_dict.keys():
        if key.startswith("image_encoder.blocks."):
            m = re.match(r"image_encoder\.blocks\.(\d+)\.", key)
            if m:
                blocks = max(blocks, int(m.group(1)) + 1)
    arch = _ARCH_BY_SHAPE.get((embed_dim, blocks))
    if arch is not None:
        return arch

    # 结构推断失败时退回文件名提示
    name = osp.basename(ckpt_path).lower()
    for token, value in _ARCH_BY_NAME.items():
        if token in name:
            return value
    raise ValueError(
        "无法识别的 SAM 权重结构（image_encoder 维度=%s，层数=%s）。\n"
        "请使用官方 SAM 权重 sam_vit_b / sam_vit_l / sam_vit_h。" % (embed_dim, blocks)
    )


def _pick_device(torch, use_gpu=None):
    """决定推理设备。use_gpu=None 表示自动检测。"""
    forced = os.environ.get("EISEG_SAM_DEVICE", "").strip().lower()
    if forced in ("cpu", "cuda"):
        use_gpu = forced == "cuda"
    if use_gpu is None:
        use_gpu = bool(torch.cuda.is_available())
    if use_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "选择了 GPU 运行 SAM，但当前 PyTorch 没有可用的 CUDA 设备。\n"
            "请安装 CUDA 版 PyTorch，或设置环境变量 EISEG_SAM_DEVICE=cpu。"
        )
    return "cuda" if use_gpu else "cpu"


class SAMModel:
    """SAM 权重 + 模型，等价于 PaddleSeg 的 ``EISegModel``。"""

    #: 供 ``InteractiveController`` 区分后端使用
    is_sam = True

    def __init__(self, ckpt_path, use_gpu=None):
        if not ckpt_path or not osp.exists(ckpt_path):
            raise Exception(f"未在 {ckpt_path} 找到 SAM 权重文件")
        _require_torch()
        _, sam_model_registry = _require_sam()

        import torch  # 延迟导入后一定可用

        try:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = _to_state_dict(state)
        if not isinstance(state, dict) or len(state) == 0:
            raise ValueError(f"{osp.basename(ckpt_path)} 不是有效的 SAM 权重文件")

        arch = detect_arch(state, ckpt_path)
        self.device = _pick_device(torch, use_gpu)
        self.arch = arch
        self.ckpt_path = ckpt_path
        self.name = "SAM-" + arch

        build_fn = sam_model_registry[arch]
        sam = build_fn(checkpoint=None)
        missing, unexpected = sam.load_state_dict(state, strict=False)
        if len(missing) > 0:
            raise ValueError(
                "SAM 权重与结构不匹配（%s），缺少 %d 个参数，例如 %s"
                % (arch, len(missing), list(missing)[:3])
            )
        sam.to(self.device)
        sam.eval()
        self.model = sam

    def build_predictor(self, **kwargs):
        return SAMPredictor(self.model, device=self.device, arch=self.arch)


class SAMPredictor:
    """把 SAM 的「点提示 -> 掩膜」包装成 EISeg 推理器接口。"""

    #: 供 ``InteractiveController`` 区分后端使用
    is_sam = True

    def __init__(self, sam_model, device="cuda", arch="vit_b"):
        SamPredictor, _ = _require_sam()
        self._sam = sam_model
        self.device = device
        self.arch = arch
        self._predictor = SamPredictor(sam_model)
        self._image = None  # 已编码的原图引用，用于跳过重复编码
        self._image_shape = None
        self._multimask = os.environ.get("EISEG_SAM_MULTIMASK", "0") == "1"

    # ------------------------------------------------------------------ 图像
    def set_input_image(self, image):
        """设置待标注图像（RGB uint8）。同一张图重复调用不会重新编码。"""
        if image is self._image and self._image_shape is not None:
            return
        img = np.asarray(image)
        if img.ndim == 2:
            img = np.repeat(img[:, :, None], 3, axis=2)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[:, :, :3]
        if img.dtype != np.uint8:
            img = img.astype(np.float32)
            if img.max() <= 1.0 + 1e-6:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        img = np.ascontiguousarray(img)
        self._predictor.set_image(img)
        self._image = image
        self._image_shape = img.shape[:2]

    # ------------------------------------------------------------------ 推理
    def get_prediction(self, clicker, prev_mask=None):
        """根据 clicker 中的正/负点返回前景概率图（0/1 的 float32）。"""
        if self._image_shape is None:
            raise RuntimeError("SAM 尚未设置图像")
        h, w = self._image_shape
        clicks = clicker.get_clicks()
        if not clicks:
            return np.zeros((h, w), dtype=np.float32)

        pos = [c for c in clicks if c.is_positive]
        if not pos:
            # 只有负点时 SAM 没有正样本，返回空掩膜，避免整幅图被选中
            return np.zeros((h, w), dtype=np.float32)

        # clicker 中存的是 (y, x)，SAM 需要 (x, y)
        coords = np.array([[c.coords[1], c.coords[0]] for c in clicks], dtype=np.float32)
        labels = np.array([1 if c.is_positive else 0 for c in clicks], dtype=np.int32)

        multimask = self._multimask
        masks, scores, _ = self._predictor.predict(
            point_coords=coords,
            point_labels=labels,
            multimask_output=multimask,
        )
        if multimask:
            mask = masks[int(np.argmax(scores))]
        else:
            mask = masks[0]
        return mask.astype(np.float32)

    # ------------------------------------------------------------------ 状态
    def get_states(self):
        # SAM 每次都由完整的点击序列重新推理，无需保存内部状态
        return {}

    def set_states(self, states):
        # undo / redo 后由 controller 用历史概率图重绘，无需重建编码
        return
