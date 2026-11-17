"""Genetic-algorithm feature selection (fitness = validation PR-AUC minus a small size penalty)."""
import numpy as np
from sklearn.metrics import average_precision_score

default_seed = 42


def score_feature_subset(mask, model_name, score_cache, light_models, x_train, y_train, x_val, y_val, penalty=0.02):
    """Fitness of one feature subset: validation PR-AUC minus a small penalty for using many features.
    A light version of the model is trained on the training data and scored on the validation data.
    The test set is never used here. Scores are cached so a repeated subset is not retrained."""
    key = mask.tobytes()
    if key in score_cache:
        return score_cache[key]

    if mask.sum() == 0:
        score = -1.0
    else:
        model = light_models[model_name]().fit(x_train[:, mask], y_train)
        val_probs = model.predict_proba(x_val[:, mask])[:, 1]
        score = average_precision_score(y_val, val_probs) - penalty * mask.mean()

    score_cache[key] = score
    return score


def pick_parent(population, scores, generator, tournament_size=3):
    """Tournament selection: draw a few random chromosomes and return the best one."""
    contenders = generator.choice(len(population), tournament_size, replace=False)
    winner = contenders[np.argmax(scores[contenders])]
    return population[winner]


def make_child(parent_a, parent_b, generator, mutation_rate, crossover_prob=0.8):
    """Create a child by uniform crossover (each feature comes from either parent),
    then flip a few features at random (mutation). Never returns an empty feature set."""
    if generator.random() < crossover_prob:
        take_from_a = generator.random(len(parent_a)) < 0.5
        child = np.where(take_from_a, parent_a, parent_b)
    else:
        child = parent_a.copy()

    flips = generator.random(len(child)) < mutation_rate
    child = np.logical_xor(child, flips)

    if not child.any():
        child[generator.integers(len(child))] = True
    return child


def run_genetic_algorithm(model_name, light_models, x_train, y_train, x_val, y_val,
                          population_size=12, generations=8, elite_count=2,
                          crossover_prob=0.8, penalty=0.02, seed=default_seed, verbose=True):
    """Search for a good subset of features for one model.

    A chromosome is a True/False mask over all features. The first chromosome is the full feature set,
    so the search starts from 'use everything'. Each generation keeps the best few chromosomes (elitism)
    and fills the rest with children of tournament-selected parents.

    Returns the best mask and an array with the best score, mean score and feature count per generation.
    """
    n_features = x_train.shape[1]
    generator = np.random.default_rng(seed)
    mutation_rate = 1.5 / n_features           # about 1.5 flipped features per child
    score_cache = {}

    population = generator.random((population_size, n_features)) < 0.6
    population[0, :] = True
    history = []

    for generation in range(generations + 1):
        scores = np.array([score_feature_subset(chromosome, model_name, score_cache, light_models,
                                                x_train, y_train, x_val, y_val, penalty)
                           for chromosome in population])
        best_index = scores.argmax()
        history.append((scores.max(), scores.mean(), population[best_index].sum()))
        if verbose:
            print(f"  [{model_name}] generation {generation:02d}: best={scores.max():.4f} "
                  f"mean={scores.mean():.4f} features={population[best_index].sum()}")

        if generation == generations:
            break

        ranking = np.argsort(-scores)
        next_population = [population[i].copy() for i in ranking[:elite_count]]
        while len(next_population) < population_size:
            parent_a = pick_parent(population, scores, generator)
            parent_b = pick_parent(population, scores, generator)
            next_population.append(make_child(parent_a, parent_b, generator, mutation_rate, crossover_prob))
        population = np.array(next_population)

    return population[best_index], np.array(history)
