# ------------------------------------------------------------------------
# Advancing Out-of-Distribution Detection via Local Neuroplasticity
# Copyright (c) 2024 Alessandro Canevaro. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from TabMedOOD (https://github.com/mazizmalayeri/TabMedOOD/)
# Copyright (c) 2023 Mohammad Azizmalayeri. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import random
import os
import torch

def set_all_seeds(seed):
  """
  Set random seeds for reproducibility across multiple libraries and devices.

  Parameters:
  -----------
  seed: int
      The random seed to use for setting all random number generators.

  Example:
  --------
  >>> set_all_seeds(42)
  """
  
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
