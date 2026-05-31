import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path

OUTPUTS_DIR = Path(__file__).resolve().parents[2] / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)


def plot_roc_curves(results: dict, X_test, y_test) -> None:
    """
    Plot ROC curves for all models on one figure.

    Parameters
    ----------
    results : dict
        Dictionary of model results containing 'model' and 'auc' keys.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    """
    plt.figure(figsize=(10, 7))

    for name, result in results.items():
        model = result["model"]
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves — All Models")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "roc_curves.png", dpi=150)
    # plt.show()


def plot_confusion_matrix(model, X_test, y_test, model_name: str) -> None:
    """
    Plot confusion matrix for the best model.

    Parameters
    ----------
    model : object
        Trained model with predict method.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    model_name : str
        Name of the model for the plot title.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Treatment", "Treatment"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    # plt.show()


def plot_learning_curve(loss_history: list, model_name: str) -> None:
    """
    Plot loss history across epochs for LogisticRegression scratch.

    Parameters
    ----------
    loss_history : list
        List of loss values per epoch.
    model_name : str
        Name of the model for the plot title.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(loss_history) + 1), loss_history, color="steelblue")
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.title(f"Learning Curve — {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "learning_curve.png", dpi=150)
    # plt.show()


def plot_feature_importance(results: dict, feature_names: list) -> None:
    """
    Plot feature importance for Random Forest and XGBoost side by side.

    Parameters
    ----------
    results : dict
        Dictionary of model results containing 'model' key.
    feature_names : list
        List of feature names.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
        model = results[name]["model"]
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]  # top 10

        ax.barh(
            [feature_names[i] for i in indices[::-1]],
            importances[indices[::-1]],
            color="steelblue"
        )
        ax.set_title(f"Feature Importance — {name}")
        ax.set_xlabel("Importance Score")

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "feature_importance.png", dpi=150)
    # plt.show()


def plot_cv_results(results: dict) -> None:
    """
    Bar chart of mean CV AUC ± std per model.

    Parameters
    ----------
    results : dict
        Dictionary of model results containing 'cv_mean' and 'cv_std' keys.
    """
    names = list(results.keys())
    means = [results[n]["cv_mean"] for n in names]
    stds = [results[n]["cv_std"] for n in names]

    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)
    plt.ylim(0.5, 1.0)
    plt.ylabel("CV AUC")
    plt.title("Cross-Validation AUC ± Std — All Models")
    plt.xticks(rotation=15)

    for bar, mean in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{mean:.4f}",
            ha="center", va="bottom", fontsize=9
        )

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "cv_results.png", dpi=150)
    # plt.show()


def plot_benchmark_summary(results: dict) -> None:
    """
    Print and save benchmark summary table as a plot.

    Parameters
    ----------
    results : dict
        Dictionary of model results.
    """
    names = list(results.keys())
    train_aucs = [results[n]["train_auc"] for n in names]
    cv_aucs = [results[n]["cv_mean"] for n in names]
    test_aucs = [results[n]["test_auc"] for n in names]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis("off")

    table_data = [
        [n, f"{tr:.4f}", f"{cv:.4f}", f"{te:.4f}"]
        for n, tr, cv, te in zip(names, train_aucs, cv_aucs, test_aucs)
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=["Model", "Train AUC", "CV AUC", "Test AUC"],
        cellLoc="center",
        loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 2)
    ax.set_title("Benchmark Summary", fontsize=13, pad=20)

    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "benchmark_summary.png", dpi=150)
    # plt.show()

    # Also print to console
    print(f"\n{'═' * 60}")
    print(f"  {'Model':<25} {'Train AUC':>10} {'CV AUC':>10} {'Test AUC':>10}")
    print(f"{'═' * 60}")
    for row in table_data:
        print(f"  {row[0]:<25} {row[1]:>10} {row[2]:>10} {row[3]:>10}")
    print(f"{'═' * 60}")

def plot_all(models: dict, results: dict,
              X_train: pd.DataFrame,
              X_test: pd.DataFrame,
              y_test: pd.Series,
              best_name: str
              ) -> None:
    lr_scratch = models["Logistic Regression"]
    plot_roc_curves(results, X_test, y_test)
    plot_confusion_matrix(results[best_name]["model"], X_test, y_test, best_name)
    plot_learning_curve(lr_scratch.loss_history, "Logistic Regression")
    plot_feature_importance(results, X_train.columns.tolist())
    plot_cv_results(results)
    plot_benchmark_summary(results)