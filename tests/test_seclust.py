import math

import numpy as np
import pytest

from glass.seclust import (
    StructuralEntropyMoveEnv,
    build_structural_entropy_dataset,
    cem_node_move_search,
    cluster_graph,
    exact_minimize_structural_entropy,
    iter_restricted_growth_strings,
    multistart_se_heuristic,
    structural_entropy,
    weighted_bridge_graph,
)


def one_hot(labels):
    labels = np.asarray(labels, dtype=np.int32)
    s = np.zeros((labels.size, int(labels.max()) + 1), dtype=float)
    s[np.arange(labels.size), labels] = 1.0
    return s


def test_hard_structural_entropy_matches_jax_objective():
    jnp = pytest.importorskip("jax.numpy")
    from glass.objectives.structural_entropy import two_dimensional_structural_entropy

    adj = weighted_bridge_graph(3, 3, bridge_weight=1.0)
    labels = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)

    hard = structural_entropy(adj, labels)
    soft = float(two_dimensional_structural_entropy(jnp.array(adj), jnp.array(one_hot(labels)), is_logits=False))

    assert math.isclose(hard, soft, rel_tol=1e-6, abs_tol=1e-6)


def test_exact_search_enumerates_bell_number_for_four_nodes():
    adj = weighted_bridge_graph(2, 2, bridge_weight=1.0)
    result = exact_minimize_structural_entropy(adj, max_nodes=4)

    assert result.partitions_evaluated == 15
    assert math.isclose(result.entropy, structural_entropy(adj, result.labels), abs_tol=1e-12)


def test_restricted_growth_max_clusters_filter():
    labelings = list(iter_restricted_growth_strings(4, max_clusters=2))

    assert len(labelings) == 8
    assert all(len(np.unique(labels)) <= 2 for labels in labelings)


def test_dataset_is_exact_labeled():
    dataset = build_structural_entropy_dataset(max_nodes=9)

    assert dataset
    for item in dataset:
        exact = exact_minimize_structural_entropy(item.adjacency, max_nodes=9)
        assert math.isclose(item.best_entropy, exact.entropy, abs_tol=1e-12)
        assert np.array_equal(item.best_labels, exact.labels)


def test_high_level_auto_uses_exact_for_small_graph():
    adj = weighted_bridge_graph(3, 3, bridge_weight=1.0)
    result = cluster_graph(adj, mode="auto", exact_max_nodes=9)

    assert result.exact
    assert result.method == "exact-rgs"


def test_heuristic_returns_valid_partition():
    adj = weighted_bridge_graph(5, 5, bridge_weight=1.0)
    result = multistart_se_heuristic(adj, starts=4, max_passes=4, seed=7)

    assert result.labels.shape == (10,)
    assert np.min(result.labels) == 0
    assert math.isclose(result.entropy, structural_entropy(adj, result.labels), abs_tol=1e-12)


def test_rl_environment_and_cem_search_are_wired():
    adj = weighted_bridge_graph(3, 3, bridge_weight=1.0)
    env = StructuralEntropyMoveEnv(adj, np.arange(adj.shape[0]))
    actions = env.legal_actions()
    labels, reward, done = env.step(actions[0])
    result = cem_node_move_search(adj, episodes=2, horizon=4, seed=3)

    assert labels.shape == (6,)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert result.labels.shape == (6,)
    assert math.isclose(result.entropy, structural_entropy(adj, result.labels), abs_tol=1e-12)
