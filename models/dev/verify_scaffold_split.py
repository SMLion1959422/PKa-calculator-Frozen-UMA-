import numpy as np
import joblib

README_CLAIMS = {
    "5-fold CV (n=5360)":            0.673,
    "scaffold split (n=1072)":       0.994,
    "Novartis external (n=263)":     1.170,
    "AvLiLuMoVe external (n=122)":   0.696,
}

def evaluate(model, X, y, label):
    pred = model.predict(X)
    mae = float(np.mean(np.abs(pred - y)))
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    r2 = 1 - np.sum((y - pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"\n{label}")
    print(f"  n = {len(y)}")
    print(f"  MAE  = {mae:.3f}")
    print(f"  RMSE = {rmse:.3f}")
    print(f"  R^2  = {r2:.3f}")
    return mae

if __name__ == "__main__":
    print("Loading model...")
    model = joblib.load("pka_model_extended.pkl")

    print("Loading arrays...")
    Xs, ys = np.load("Xs.npy"), np.load("ys.npy")
    Xe, ye = np.load("Xe.npy"), np.load("ye.npy")

    mae_s = evaluate(model, Xs, ys, "Xs.npy / ys.npy  (suspected: scaffold split)")
    mae_e = evaluate(model, Xe, ye, "Xe.npy / ye.npy  (suspected: extended/training set)")

    print("\n" + "=" * 70)
    print("README's claimed numbers, for comparison:")
    for label, mae in README_CLAIMS.items():
        print(f"  {label:<32} MAE = {mae:.3f}")

    print("\n" + "=" * 70)
    print("Interpretation:")
    print(f"  - If Xs/ys MAE ({mae_s:.3f}) is close to the scaffold-split claim")
    print(f"    (0.994) -> strong evidence Xs/ys IS the scaffold-split eval set")
    print(f"    and pka_model_extended.pkl IS the documented model.")
    print(f"  - If Xe/ye MAE ({mae_e:.3f}) is much LOWER than any README number")
    print(f"    -> Xe/ye is likely data the model was TRAINED on (fit, not")
    print(f"    generalization) - expected and not concerning on its own.")
    print(f"  - If Xs/ys MAE is wildly different from 0.994 (e.g. <0.3 or >2)")
    print(f"    -> Xs/ys probably isn't the scaffold-split test set, or this")
    print(f"    isn't the model that produced that number.")
