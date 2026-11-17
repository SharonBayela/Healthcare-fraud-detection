"""Provider-level bootstrap comparison of PR-AUC with Holm correction."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

default_seed = 42


def bootstrap_pr_auc(probs_by_model, y_true, provider_ids, n_rounds=300, seed=default_seed):
    """Bootstrap PR-AUC for every model on the same resamples of PROVIDERS.

    Each round draws providers with replacement (taking all of their claims), so claims from the same
    provider stay together. Using the same resample for all models makes the comparison paired.
    Returns a dict: model key -> array of PR-AUC values.
    """
    claim_positions = pd.Series(np.arange(len(provider_ids))).groupby(provider_ids).indices
    unique_providers = np.array(list(claim_positions.keys()))
    generator = np.random.default_rng(seed)
    scores = {key: [] for key in probs_by_model}

    for _ in range(n_rounds):
        drawn = generator.choice(unique_providers, len(unique_providers), replace=True)
        rows = np.concatenate([claim_positions[provider] for provider in drawn])
        if y_true[rows].min() == y_true[rows].max():
            continue
        for key, probs in probs_by_model.items():
            scores[key].append(average_precision_score(y_true[rows], probs[rows]))

    return {key: np.array(values) for key, values in scores.items()}


def compare_two_models(boot_scores, key_a, key_b):
    """Difference in PR-AUC (A minus B): mean, 95% interval and a two-sided bootstrap p-value."""
    difference = boot_scores[key_a] - boot_scores[key_b]
    low, high = np.percentile(difference, [2.5, 97.5])
    p_value = min(1.0, 2 * min((difference <= 0).mean(), (difference >= 0).mean()))
    return {"A": f"{key_a[0]} [{key_a[1]}]", "B": f"{key_b[0]} [{key_b[1]}]",
            "dPR-AUC (A-B)": difference.mean(), "CI low": low, "CI high": high,
            "p (bootstrap)": p_value, "CI excludes 0": bool(low > 0 or high < 0)}


def holm_adjust(p_values):
    """Holm step-down correction for running several comparisons at once."""
    p_values = np.asarray(p_values, dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values))
    running_max = 0.0
    for rank, index in enumerate(order):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def default_pairs(model_names):
    """The comparisons used in the project: GA vs All for each model, then model against model."""
    pairs = [((name, "GA"), (name, "All")) for name in model_names]
    for variant in ["All", "GA"]:
        for other in ["Logistic Regression", "Random Forest", "Neural Network (MLP)"]:
            pairs.append((("Bayesian Logistic Regression", variant), (other, variant)))
        pairs.append((("Neural Network (MLP)", variant), ("Random Forest", variant)))
        pairs.append((("Neural Network (MLP)", variant), ("Logistic Regression", variant)))
    return pairs


def build_comparison_table(boot_scores, pairs):
    """Compare every pair, then add Holm-adjusted p-values and a significance flag."""
    table = pd.DataFrame([compare_two_models(boot_scores, a, b) for a, b in pairs])
    n_valid_rounds = len(next(iter(boot_scores.values())))
    table["p (bootstrap)"] = table["p (bootstrap)"].clip(lower=1 / (n_valid_rounds + 1))
    table["p (Holm-adjusted)"] = holm_adjust(table["p (bootstrap)"].values)
    table["Significant after correction"] = table["p (Holm-adjusted)"] < 0.05
    return table
