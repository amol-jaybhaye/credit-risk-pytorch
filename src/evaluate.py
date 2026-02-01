import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve,
    confusion_matrix, ConfusionMatrixDisplay,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    KS = max difference between cumulative distributions of scores
    for positives vs negatives.
    """
    y_true = y_true.astype(int)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    thresholds = np.unique(y_score)
    thresholds.sort()
    # compute empirical CDFs
    pos_cdf = np.searchsorted(np.sort(pos), thresholds, side="right") / len(pos)
    neg_cdf = np.searchsorted(np.sort(neg), thresholds, side="right") / len(neg)
    return float(np.max(np.abs(pos_cdf - neg_cdf)))

def eval_and_plots(y_true, y_prob, threshold: float, out_dir):
    out_dir = str(out_dir)
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    metrics = {}
    metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
    metrics["ks"] = float(ks_statistic(y_true, y_prob))
    metrics["brier"] = float(brier_score_loss(y_true, y_prob))

    # Confusion matrix
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/confusion_matrix.png", dpi=140)
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/roc_curve.png", dpi=140)
    plt.close()

    # PR curve
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/pr_curve.png", dpi=140)
    plt.close()

    # Calibration curve (reliability diagram)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy="uniform")
    plt.figure()
    plt.plot(prob_pred, prob_true, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/calibration_curve.png", dpi=140)
    plt.close()

    return metrics
