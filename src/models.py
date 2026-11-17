"""Models used in the project: Bayesian logistic regression, an MLP ensemble, and builders for the baselines."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

default_seed = 42
paper_random_state = 50   # the paper used random_state=50

model_names = ["Logistic Regression", "Random Forest", "Bayesian Logistic Regression", "Neural Network (MLP)"]


class BayesianLogisticRegression(BaseEstimator, ClassifierMixin):
    """Logistic regression with a normal prior on the weights and a Laplace approximation of the posterior.

    predict_proba averages predictions over weight samples from the posterior;
    predict_with_spread also returns their standard deviation (per-claim uncertainty).
    """

    def __init__(self, prior_sd=1.0, n_samples=200, random_state=default_seed):
        self.prior_sd = prior_sd
        self.n_samples = n_samples
        self.random_state = random_state

    def add_intercept(self, x):
        return np.hstack([np.ones((len(x), 1)), x])

    def fit(self, x, y):
        x = self.add_intercept(np.asarray(x, dtype=float))
        y = np.asarray(y, dtype=float)
        n_weights = x.shape[1]
        prior_precision = np.full(n_weights, 1.0 / self.prior_sd ** 2)
        prior_precision[0] = 1.0 / 100.0          # very wide prior on the intercept

        def negative_log_posterior(weights):
            scores = x @ weights
            return np.sum(np.logaddexp(0, scores) - y * scores) + 0.5 * np.sum(prior_precision * weights * weights)

        def gradient(weights):
            return x.T @ (expit(x @ weights) - y) + prior_precision * weights

        best = minimize(negative_log_posterior, np.zeros(n_weights), jac=gradient,
                        method="L-BFGS-B", options={"maxiter": 300})
        probabilities = expit(x @ best.x)
        curvature = (x * (probabilities * (1 - probabilities))[:, None]).T @ x + np.diag(prior_precision)
        self.weights_ = best.x
        self.covariance_ = np.linalg.inv(curvature)
        self.classes_ = np.array([0, 1])
        return self

    def predict_with_spread(self, x, chunk_size=20000):
        """Return (mean predicted fraud probability, standard deviation) for each row."""
        x = np.asarray(x, dtype=float)
        generator = np.random.default_rng(self.random_state)
        weight_samples = generator.multivariate_normal(self.weights_, self.covariance_, size=self.n_samples)
        means, spreads = [], []
        for start in range(0, len(x), chunk_size):
            chunk = self.add_intercept(x[start:start + chunk_size])
            sampled = expit(chunk @ weight_samples.T)
            means.append(sampled.mean(axis=1))
            spreads.append(sampled.std(axis=1))
        return np.concatenate(means), np.concatenate(spreads)

    def predict_proba(self, x):
        mean_prob, _ = self.predict_with_spread(x)
        return np.column_stack([1 - mean_prob, mean_prob])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


class MLPEnsemble(BaseEstimator, ClassifierMixin):
    """Several small MLPs trained from different random starts.
    Mean prediction = fraud probability; standard deviation across members = uncertainty."""

    def __init__(self, n_members=3, hidden_layers=(64, 32), alpha=1e-3, max_iter=60, random_state=default_seed):
        self.n_members = n_members
        self.hidden_layers = hidden_layers
        self.alpha = alpha
        self.max_iter = max_iter
        self.random_state = random_state

    def fit(self, x, y):
        self.members_ = []
        for member_number in range(self.n_members):
            network = MLPClassifier(hidden_layer_sizes=self.hidden_layers, alpha=self.alpha,
                                    early_stopping=True, n_iter_no_change=5, max_iter=self.max_iter,
                                    random_state=self.random_state + member_number)
            self.members_.append(network.fit(x, y))
        self.classes_ = np.array([0, 1])
        return self

    def member_probabilities(self, x):
        return np.stack([member.predict_proba(x)[:, 1] for member in self.members_])

    def predict_proba(self, x):
        mean_prob = self.member_probabilities(x).mean(axis=0)
        return np.column_stack([1 - mean_prob, mean_prob])

    def predict(self, x):
        return (self.predict_proba(x)[:, 1] >= 0.5).astype(int)


def build_logistic_regression(max_iter=1000, class_weight="balanced"):
    """Paper settings: L1 penalty, C=0.03, balanced class weights, liblinear solver."""
    return LogisticRegression(penalty="l1", C=0.03, class_weight=class_weight, solver="liblinear",
                              random_state=paper_random_state, max_iter=max_iter)


def build_random_forest(n_trees, class_weight="balanced"):
    """Paper settings: depth 4, balanced class weights."""
    return RandomForestClassifier(n_estimators=n_trees, max_depth=4, class_weight=class_weight,
                                  n_jobs=-1, random_state=paper_random_state)


def make_full_models(rf_trees=200):
    """Full versions of the four models, used for the final results."""
    return {
        "Logistic Regression": lambda: build_logistic_regression(),
        "Random Forest": lambda: build_random_forest(rf_trees),
        "Bayesian Logistic Regression": lambda: BayesianLogisticRegression(prior_sd=1.0, n_samples=200),
        "Neural Network (MLP)": lambda: MLPEnsemble(n_members=3, max_iter=60),
    }


def make_light_models():
    """Lighter versions, used only inside the genetic algorithm to keep it fast."""
    return {
        "Logistic Regression": lambda: build_logistic_regression(300),
        "Random Forest": lambda: build_random_forest(60),
        "Bayesian Logistic Regression": lambda: BayesianLogisticRegression(prior_sd=1.0, n_samples=50),
        "Neural Network (MLP)": lambda: MLPEnsemble(n_members=2, max_iter=40),
    }
