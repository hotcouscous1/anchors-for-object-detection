import torch
import numpy as np
import math
from typing import List, Tuple, Dict, OrderedDict

Numpy = np.array
Tensor = torch.Tensor

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

