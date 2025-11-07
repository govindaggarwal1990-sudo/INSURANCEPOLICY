
import numpy as np
import pandas as pd
from typing import Dict
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt

def prepare_data(df: pd.DataFrame, target_col: str):
    y_raw = df[target_col].astype(str)
    X = df.drop(columns=[target_col]).copy()
    # light NA handling to avoid crashes
    for c in X.columns:
        if X[c].dtype.kind in "biufc":
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(X[c].median())
        else:
            X[c] = X[c].fillna("NA").astype(str)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    classes = list(le.classes_)
    return X, y, classes

def _build_preprocessor(X: pd.DataFrame):
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

def _models(random_state=42):
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
        "Gradient Boosting": GradientBoostingClassifier(random_state=random_state),
    }

def _plot_confusion(cm, title, labels):
    fig, ax = plt.subplots(figsize=(4.6, 4.3), dpi=130)
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(labels)), yticks=np.arange(len(labels)),
           xticklabels=labels, yticklabels=labels,
           ylabel="True label", xlabel="Predicted label", title=title)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    return fig

def evaluate_models(X: pd.DataFrame, y: np.ndarray, classes, test_size=0.2, random_state=42, cv_splits=5):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    preproc = _build_preprocessor(X_train)
    models = _models(random_state=random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    records = []
    plots: Dict[str, plt.Figure] = {}
    fitted_pipelines = {}
    roc_store = {}

    # for feature names later
    num_cols = X_train.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()

    for name, clf in models.items():
        pipe = Pipeline(steps=[("prep", preproc), ("clf", clf)])
        auc_cv = cross_val_score(pipe, X, y, scoring="roc_auc", cv=cv, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe

        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)

        if hasattr(pipe.named_steps["clf"], "predict_proba"):
            y_proba_test = pipe.predict_proba(X_test)[:, 1]
        elif hasattr(pipe.named_steps["clf"], "decision_function"):
            s = pipe.decision_function(X_test)
            y_proba_test = (s - s.min()) / (s.max() - s.min() + 1e-9)
        else:
            y_proba_test = y_pred_test.astype(float)

        rec = {
            "Algorithm": name,
            "Training Accuracy": round(accuracy_score(y_train, y_pred_train), 4),
            "Testing Accuracy": round(accuracy_score(y_test, y_pred_test), 4),
            "Precision (test)": round(precision_score(y_test, y_pred_test, zero_division=0), 4),
            "Recall (test)": round(recall_score(y_test, y_pred_test, zero_division=0), 4),
            "F1-score (test)": round(f1_score(y_test, y_pred_test, zero_division=0), 4),
            "ROC AUC (CV mean)": round(auc_cv.mean(), 4),
            "ROC AUC (test)": round(roc_auc_score(y_test, y_proba_test), 4),
        }
        records.append(rec)

        train_cm = confusion_matrix(y_train, y_pred_train)
        test_cm  = confusion_matrix(y_test,  y_pred_test)
        plots[f"{name} — Training Confusion Matrix"] = _plot_confusion(train_cm, f"{name} — Training Confusion Matrix", classes)
        plots[f"{name} — Testing Confusion Matrix"]  = _plot_confusion(test_cm,  f"{name} — Testing Confusion Matrix", classes)

        fpr, tpr, _ = roc_curve(y_test, y_proba_test)
        roc_store[name] = (fpr, tpr, rec["ROC AUC (test)"])

        if hasattr(pipe.named_steps["clf"], "feature_importances_"):
            ohe = pipe.named_steps["prep"].named_transformers_.get("cat", None)
            cat_names = ohe.get_feature_names_out(cat_cols).tolist() if (ohe and len(cat_cols)>0) else []
            feat_names = cat_names + num_cols
            importances = pipe.named_steps["clf"].feature_importances_
            order = np.argsort(importances)[::-1][:15]
            fig, ax = plt.subplots(figsize=(8.4, 6.5), dpi=140)
            ax.barh([feat_names[i] for i in order][::-1], importances[order][::-1])
            ax.set_title(f"Top 15 Feature Importances — {name}")
            ax.set_xlabel("Importance"); ax.set_ylabel("Feature"); plt.tight_layout()
            plots[f"{name} — Feature Importances"] = fig

    fig_roc, ax = plt.subplots(figsize=(6.6, 5.6), dpi=140)
    for name, (fpr, tpr, aucv) in roc_store.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC={aucv:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate"); ax.set_title("ROC Curves — Test Set")
    ax.legend(loc="lower right"); plt.tight_layout()
    plots["ROC Curves — All Models"] = fig_roc

    results_df = pd.DataFrame.from_records(records).set_index("Algorithm")
    return results_df, plots, fitted_pipelines

def run_inference_on_new_data(new_df: pd.DataFrame, fitted_pipeline, proba_colname="Attrition_Prob"):
    Xnew = new_df.copy()
    for c in Xnew.columns:
        if Xnew[c].dtype.kind in "biufc":
            Xnew[c] = pd.to_numeric(Xnew[c], errors="coerce").fillna(Xnew[c].median())
        else:
            Xnew[c] = Xnew[c].fillna("NA").astype(str)
    preds = fitted_pipeline.predict(Xnew)
    if hasattr(fitted_pipeline.named_steps["clf"], "predict_proba"):
        proba = fitted_pipeline.predict_proba(Xnew)[:, 1]
    elif hasattr(fitted_pipeline.named_steps["clf"], "decision_function"):
        s = fitted_pipeline.decision_function(Xnew)
        proba = (s - s.min()) / (s.max() - s.min() + 1e-9)
    else:
        proba = preds.astype(float)
    out = new_df.copy()
    out["Predicted_Attrition"] = preds
    out[proba_colname] = proba
    return preds, proba, out
