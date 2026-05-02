import random

import numpy as np
import torch


def set_seed(seed=42):
    # 1. Python 内置随机模块
    random.seed(seed)
    # 2. NumPy 随机种子
    np.random.seed(seed)
    # 3. PyTorch CPU 随机种子
    torch.manual_seed(seed)
    # 4. PyTorch GPU 随机种子（单卡/多卡）
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多卡场景也生效
    # 5. 禁用 PyTorch 的自动优化算法（保证确定性）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
