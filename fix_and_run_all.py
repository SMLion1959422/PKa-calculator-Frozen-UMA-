import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

print("=== 1. Inspecting and fixing training pipeline ===")
# Read the existing training script to grab data loading and model architecture
with open('train_v19_pretrained.py', 'r', encoding='utf-8') as f:
    code = f.read()

# Replace L1Loss with MSELoss for smoother continuous gradients
fixed_code = code.replace('l1=nn.L1Loss()', 'l1=nn.MSELoss()')
# Replace IsotonicRegression with smooth Ridge calibration
fixed_code = fixed_code.replace('cal=IsotonicRegression(out_of_bounds="clip").fit(oof,yc)', 'cal=Ridge(alpha=1.0).fit(oof.reshape(-1, 1), yc)')
fixed_code = fixed_code.replace('pred = float(cal.predict([raw])[0])', 'pred = float(cal.predict(np.array([[raw]]))[0])')

with open('train_v19_smooth.py', 'w', encoding='utf-8') as f:
    f.write(fixed_code)

print("=== 2. Running smoothed training and evaluation script ===")
os.system('python train_v19_smooth.py')
