# Biomarker Prioritization & Risk Classification Pipeline

**Live app:** https://biomarker-risk-pipeline.streamlit.app/
**Live 3D demo:** https://vajrapuharshitha.github.io/biomarker-risk-pipeline/

Supervised ML pipeline that ranks candidate biomarkers and classifies disease risk, with clinical thresholding.
Built for the "Cancer & Atherosclerosis Biomarkers" case-study theme (AI feature selection and risk scoring).

**Dataset:** Wisconsin Diagnostic Breast Cancer (569 samples, 30 numeric features), bundled with scikit-learn. Positive class = malignant.
This is a public demonstration dataset; swap in your own biomarker table by editing the data-loading block.

## Method
1. Stratified 75/25 train/hold-out split (n=426 / n=143).
2. Leakage-safe pipelines: scaling and mutual-information feature selection (top 10) run inside 5-fold CV.
3. Compare Logistic Regression, Random Forest, XGBoost by cross-validated ROC-AUC (training set only).
4. Evaluate on the untouched hold-out set; plot ROC curves.
5. Clinical thresholding: choose the highest threshold reaching >=95% sensitivity on the training set, then report hold-out sensitivity/specificity.
6. Rank selected biomarkers by model importance.

## Results (seed 42)
| Model | CV AUC (mean +/- SD) | Hold-out AUC |
|---|---|---|
| Logistic Regression | 0.986 +/- 0.006 | 0.998 |
| Random Forest | 0.981 +/- 0.010 | 0.993 |
| XGBoost | 0.977 +/- 0.013 | 0.992 |

At the selected threshold (0.262) the best model reached hold-out sensitivity 100% (53/53 malignant) and specificity 96.7% (87/90 benign).
Top biomarkers: worst concave points, worst area, worst radius, area error, worst perimeter.

Caveats: small single-source dataset, one split; results are not clinical evidence. External validation would be the next step.

## Run
```
pip install -r requirements.txt
python run_pipeline.py
```
Outputs go to `results/` (ROC plot, biomarker ranking, CV table, summary.json).
