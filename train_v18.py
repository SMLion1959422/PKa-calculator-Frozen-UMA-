"""Train v18 on the maximum-data set, same hybrid recipe as v16."""
import numpy as np, pandas as pd, joblib, lightgbm as lgb
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler

core = joblib.load("core_maxdata.pkl")
X = np.nan_to_num(np.vstack(core.vec.values)); y = core.pKa.values
print(f"training on {len(y)} molecules, dim {X.shape[1]}")
print(f"label range: {y.min():.2f} - {y.max():.2f}")

sc = StandardScaler().fit(X); Xs = sc.transform(X)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
og = np.zeros(len(y)); orr = np.zeros(len(y))
for i,(tr,va) in enumerate(kf.split(X)):
    og[va] = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                                verbose=-1, random_state=42).fit(X[tr],y[tr]).predict(X[va])
    orr[va] = RidgeCV(alphas=np.logspace(-2,4,25)).fit(Xs[tr],y[tr]).predict(Xs[va])
    print(f"  fold {i+1}/5")
bw,bm = 0.0,1e9
for w in np.arange(0,1.01,0.05):
    m = np.mean(np.abs((1-w)*og + w*orr - y))
    if m < bm: bm,bw = m,w
bl = (1-bw)*og + bw*orr
cal = IsotonicRegression(out_of_bounds="clip").fit(bl,y)
print(f"\nOOF calibrated: {np.mean(np.abs(cal.predict(bl)-y)):.3f}  (v16 was 0.485, blend w={bw:.2f})")

gf = lgb.LGBMRegressor(n_estimators=600, learning_rate=0.05, num_leaves=31,
                        verbose=-1, random_state=42).fit(X,y)
rf = RidgeCV(alphas=np.logspace(-2,4,25)).fit(Xs,y)
joblib.dump({"gbm":gf,"ridge":rf,"scaler":sc,"blend_w":bw,"calibrator":cal},
            "models/model_core_v18_maxdata.pkl")
print("saved -> models/model_core_v18_maxdata.pkl")
