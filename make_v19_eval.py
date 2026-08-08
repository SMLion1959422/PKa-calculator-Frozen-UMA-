import re
src=open("eval_v16.py",encoding="utf-8").read()
src=src.replace('b = joblib.load("models/model_core_v16_elec.pkl")\ngbm, ridge, scaler, bw, cal = b["gbm"], b["ridge"], b["scaler"], b["blend_w"], b["calibrator"]',
'''import torch, torch.nn as nn
class Head(nn.Module):
    def __init__(s,d,h=512):
        super().__init__()
        s.net=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Dropout(0.2),
                            nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(s,x): return s.net(x).squeeze(-1)
b = joblib.load("models/model_core_v19_pretrained.pkl")
_m = Head(b["dim"], b["hidden"])
_m.load_state_dict({k: torch.tensor(v) for k,v in b["state_dict"].items()})
_m.eval()
scaler, cal = b["scaler"], b["calibrator"]''')
src=src.replace('raw = (1-bw)*gbm.predict(feat)[0] + bw*ridge.predict(scaler.transform(feat))[0]',
'''with torch.no_grad():
                raw = float(_m(torch.tensor(scaler.transform(feat), dtype=torch.float32)).item())''')
src=src.replace('raw = (1-bw)*gbm.predict(feat)[0] + bw*ridge.predict(sc.transform(feat))[0]',
'''with torch.no_grad():
                raw = float(_m(torch.tensor(scaler.transform(feat), dtype=torch.float32)).item())''')
src=src.replace("characterization_external_v16.csv","characterization_external_v19.csv")
src=src.replace("=== v16: UMA + ELECTRONIC","=== v19: + noisy-site pretraining")
open("eval_v19.py","w",encoding="utf-8").write(src)
print("wrote eval_v19.py")
