import re
src = open("eval_core_v9_marvin.py", encoding="utf-8").read()
src = src.replace("models/model_core_v7_clean.pkl", "models/model_core_v10_matched.pkl")
src = src.replace("characterization_external_v9_marvin.csv",
                  "characterization_external_v10.csv")
src = src.replace("=== v9: MARVIN ground-truth sites", "=== v10: MATCHED-SITE training + Marvin sites")
open("eval_core_v10.py", "w", encoding="utf-8").write(src)
print("wrote eval_core_v10.py")
