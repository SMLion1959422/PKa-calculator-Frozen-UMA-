import numpy as np
import joblib

models = [
    "pka_model_extended.pkl",
    "pka_model_augmented.pkl",
    "model_sequential.pkl",
    "model_sequential_fixed.pkl",
    "model_sequential_corrected.pkl",
]
arrays = [("Xs.npy", "ys.npy"), ("Xe.npy", "ye.npy")]

def evaluate(model, X, y):
    try:
        pred = model.predict(X)
    except Exception as e:
        return None, None, f"predict() failed: {e}"
    mae = float(np.mean(np.abs(pred - y)))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    return mae, r2, None

print(f"{'model':<32} {'array':<10} {'n':>6} {'MAE':>8} {'R^2':>8}")
print("-" * 70)
for mpath in models:
    try:
        model = joblib.load(mpath)
    except Exception as e:
        print(f"{mpath:<32} FAILED TO LOAD: {e}")
        continue
    for xf, yf in arrays:
        X, y = np.load(xf), np.load(yf)
        mae, r2, err = evaluate(model, X, y)
        if err:
            print(f"{mpath:<32} {xf:<10} {len(y):>6} {err}")
        else:
            print(f"{mpath:<32} {xf:<10} {len(y):>6} {mae:>8.3f} {r2:>8.3f}")
