"""
Biomarker prioritization and risk classification pipeline.

Dataset : Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric features),
          bundled with scikit-learn. Label: malignant (1) vs benign (0).
Steps   : stratified hold-out split -> leakage-safe pipelines (scaling + feature
          selection inside CV) -> model comparison by cross-validated ROC-AUC ->
          hold-out evaluation -> clinical thresholding -> biomarker ranking.
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier

SEED, K_FEATURES = 42, 10

# 1. Data. sklearn encodes malignant as 0, so flip to make malignant the positive class.
data = load_breast_cancer(as_frame=True)
X, y = data.data, 1 - data.target
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)

# 2. Models. Selection sits inside each pipeline so CV never sees held-out folds.
def make(model):
    return Pipeline([("scale", StandardScaler()),
                     ("select", SelectKBest(mutual_info_classif, k=K_FEATURES)),
                     ("clf", model)])

models = {
    "Logistic Regression": make(LogisticRegression(max_iter=2000, random_state=SEED)),
    "Random Forest": make(RandomForestClassifier(n_estimators=300, random_state=SEED)),
    "XGBoost": make(XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                                  eval_metric="logloss", random_state=SEED)),
}

# 3. Cross-validated comparison on the training set only.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = []
for name, pipe in models.items():
    s = cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="roc_auc")
    cv_rows.append({"model": name, "cv_auc_mean": round(s.mean(), 4), "cv_auc_sd": round(s.std(), 4)})
cv_df = pd.DataFrame(cv_rows).sort_values("cv_auc_mean", ascending=False)
best_name = cv_df.iloc[0]["model"]
print(cv_df.to_string(index=False))

# 4. Hold-out evaluation for every model (best model flagged), plus ROC plot.
fig, ax = plt.subplots(figsize=(5, 4.5))
holdout = {}
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    holdout[name] = round(roc_auc_score(y_te, p), 4)
    fpr, tpr, _ = roc_curve(y_te, p)
    ax.plot(fpr, tpr, label=f"{name} (AUC {holdout[name]:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Hold-out ROC (n=%d)" % len(y_te))
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("results/roc_curves.png", dpi=150); plt.close(fig)

# 5. Clinical thresholding for the best model: pick the highest threshold that still
#    reaches >=95% sensitivity, chosen on training-set CV-free predictions to avoid
#    tuning on the hold-out set, then report hold-out performance at that threshold.
best = models[best_name]
p_tr = best.predict_proba(X_tr)[:, 1]
fpr_tr, tpr_tr, thr_tr = roc_curve(y_tr, p_tr)
ok = np.where(tpr_tr >= 0.95)[0]
thr = float(thr_tr[ok[0]])
p_te = best.predict_proba(X_te)[:, 1]
pred = (p_te >= thr).astype(int)
tn, fp, fn, tp = confusion_matrix(y_te, pred).ravel()
clinical = {"threshold": round(thr, 4), "sensitivity": round(tp / (tp + fn), 4),
            "specificity": round(tn / (tn + fp), 4), "tp": int(tp), "fn": int(fn),
            "tn": int(tn), "fp": int(fp)}
print("Clinical threshold:", clinical)

# 6. Biomarker prioritization: features kept by selection, ranked by the best
#    tree/linear model's importance on the training set.
sel = best.named_steps["select"]
kept = X.columns[sel.get_support()]
clf = best.named_steps["clf"]
imp = (np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_)
rank = pd.DataFrame({"biomarker": kept, "importance": imp}).sort_values("importance", ascending=False)
rank.to_csv("results/biomarker_ranking.csv", index=False)
fig, ax = plt.subplots(figsize=(6, 4))
ax.barh(rank["biomarker"][::-1], rank["importance"][::-1], color="#1F3A5F")
ax.set(xlabel="Importance", title=f"Top {K_FEATURES} biomarkers ({best_name})")
fig.tight_layout(); fig.savefig("results/biomarker_importance.png", dpi=150); plt.close(fig)

cv_df.to_csv("results/cv_model_comparison.csv", index=False)
json.dump({"n_train": len(y_tr), "n_test": len(y_te), "best_model": best_name,
           "holdout_auc": holdout, "clinical_threshold": clinical},
          open("results/summary.json", "w"), indent=2)
print("Hold-out AUC:", holdout)
print(rank.to_string(index=False))
