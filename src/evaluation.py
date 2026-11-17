"""Prediction, uncertainty and evaluation metrics."""
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, recall_score,
                             precision_score, brier_score_loss, precision_recall_curve)

from models import BayesianLogisticRegression, MLPEnsemble


def predict_with_uncertainty(model, x):
    """Return (fraud probability, uncertainty) for any of the four models.

    Bayesian LR: spread of the posterior predictions.
    Random forest: spread across the individual trees.
    Neural network ensemble: spread across the ensemble members.
    Plain logistic regression has no uncertainty, so NaN is returned for it.
    """
    if isinstance(model, BayesianLogisticRegression):
        return model.predict_with_spread(x)
    if isinstance(model, RandomForestClassifier):
        x32 = np.asarray(x, dtype=np.float32)
        tree_probs = np.stack([tree.predict_proba(x32)[:, 1] for tree in model.estimators_])
        return tree_probs.mean(axis=0), tree_probs.std(axis=0)
    if isinstance(model, MLPEnsemble):
        member_probs = model.member_probabilities(x)
        return member_probs.mean(axis=0), member_probs.std(axis=0)
    probs = model.predict_proba(x)[:, 1]
    return probs, np.full(len(probs), np.nan)


def expected_calibration_error(y_true, probs, n_bins=10):
    """Average gap between predicted probability and actual fraud rate, over equal-width bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)[1:-1]
    bin_number = np.digitize(probs, bin_edges)
    total_error = 0.0
    for b in range(n_bins):
        in_bin = bin_number == b
        if in_bin.any():
            gap = abs(y_true[in_bin].mean() - probs[in_bin].mean())
            total_error += gap * in_bin.sum() / len(y_true)
    return total_error


def best_f1_threshold(y_true, probs):
    """Probability cut-off that gives the highest F1 score. Only call this on validation data."""
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    return thresholds[np.argmax(f1_scores[:-1])]


def evaluate_predictions(y_true, probs, uncertainty, threshold):
    """Compute the evaluation metrics, plus uncertainty diagnostics when uncertainty is available."""
    predicted = (probs >= threshold).astype(int)
    scores = {
        "PR-AUC": average_precision_score(y_true, probs),
        "ROC-AUC": roc_auc_score(y_true, probs),
        "F1": f1_score(y_true, predicted),
        "Recall": recall_score(y_true, predicted),
        "Precision": precision_score(y_true, predicted, zero_division=0),
        "Brier": brier_score_loss(y_true, probs),
        "ECE": expected_calibration_error(y_true, probs),
        "Threshold": threshold,
    }
    if not np.isnan(uncertainty).any():
        is_error = (predicted != y_true).astype(int)
        most_uncertain = uncertainty >= np.quantile(uncertainty, 0.8)
        scores["Mean uncertainty"] = uncertainty.mean()
        scores["Uncertainty->error AUC"] = (roc_auc_score(is_error, uncertainty)
                                            if is_error.min() != is_error.max() else np.nan)
        scores["Err rate (top 20% uncertain)"] = is_error[most_uncertain].mean()
        scores["Err rate (rest)"] = is_error[~most_uncertain].mean()
    return scores


def fit_and_evaluate(model_name, variant, feature_columns, models, x_train, y_train, x_val, y_val, x_test, y_test):
    """Train one model on the chosen feature columns, pick its threshold on the validation set and
    score it on the test set.

    models is a dict of model name -> function that builds a fresh model (see models.make_full_models).
    Returns (scores, details). details holds the fitted model, test probabilities, test uncertainty and
    threshold, so the caller can store them under the key (model_name, variant).
    """
    start_time = time.time()
    model = models[model_name]().fit(x_train[:, feature_columns], y_train)

    val_probs, _ = predict_with_uncertainty(model, x_val[:, feature_columns])
    threshold = best_f1_threshold(y_val, val_probs)
    probs, uncertainty = predict_with_uncertainty(model, x_test[:, feature_columns])

    scores = evaluate_predictions(y_test, probs, uncertainty, threshold)
    scores.update({"Model": model_name, "Features": variant,
                   "n_features": len(feature_columns),
                   "Seconds": round(time.time() - start_time, 1)})
    details = {"model": model, "probs": probs, "uncertainty": uncertainty, "threshold": threshold}
    return scores, details
