# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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


import sys
import os
import os.path as osp
import logging
from datetime import datetime

from qtpy import QtCore
import cv2

__APPNAME__ = "EISeg"
__VERSION__ = "0.5.0"

# PyTorch 必须在 paddle 之前导入。
# Windows 上如果先 import paddle 再 import torch，torch 的 shm.dll 会因为依赖冲突
# 加载失败（OSError: [WinError 127]），SAM 后端也就用不了；反过来先导入 torch 则两者
# 都能正常工作。所以这里做一次「尽力而为」的提前导入：没装 torch 时直接跳过，
# 不影响只使用 PaddleSeg 模型的用户。设置环境变量 EISEG_NO_TORCH=1 可以关闭这一步。
TORCH_READY = False
if os.environ.get("EISEG_NO_TORCH", "") != "1":
    try:
        import torch  # noqa: F401

        TORCH_READY = True
    except Exception:  # noqa: BLE001  torch 缺失或 DLL 不可用时静默跳过
        TORCH_READY = False



pjpath = osp.dirname(osp.realpath(__file__))
sys.path.append(pjpath)

for k, v in os.environ.items():
    if k.startswith("QT_") and "cv2" in v:
        del os.environ[k]

# log
settings = QtCore.QSettings(
    osp.join(pjpath, "config/setting.ini"), QtCore.QSettings.IniFormat
)

logFolder = settings.value("logFolder")
logLevel = bool(settings.value("log"))
logDays = settings.value("logDays")

if logFolder is None or len(logFolder) == 0:
    logFolder = osp.normcase(osp.join(pjpath, "log"))
if not osp.exists(logFolder):
    os.makedirs(logFolder)

if logLevel:
    logLevel = logging.DEBUG
else:
    logLevel = logging.CRITICAL
if logDays:
    logDays = int(logDays)
else:
    logDays = 7
# TODO: 删除大于logDays 的 log

t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logger = logging.getLogger("EISeg Logger")
handler = logging.FileHandler(osp.normcase(osp.join(logFolder, f"eiseg-{t}.log")))
handler.setFormatter(
    logging.Formatter(
        "%(levelname)s - %(asctime)s - %(filename)s - %(funcName)s - %(message)s"
    )
)
logger.setLevel(logLevel)
logger.addHandler(handler)
