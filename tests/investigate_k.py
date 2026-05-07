import jax
import jax.numpy as jnp
import numpy as np
from tests.benchmark import get_karate_club, get_sbm, run_glass_jax_multistart, eval_structural_entropy, eval_modularity
from glass.objectives.structural_entropy import two_dimensional_structural_entropy
from glass.objectives.modularity import soft_modularity

def run_k_sweep(adj, name, ks=[2, 3, 4, 5, 6, 8, 10]):
    print(f"--- Dataset: {name} ---")
    print(f"{'K':<4} | {'Objective Optimized':<20} | {'Final SE (↓)':<15} | {'Final Modularity (↑)':<20}")
    print("-" * 65)
    
    for k in ks:
        # Optimize Modularity
        labels_mod, _ = run_glass_jax_multistart(adj, k, soft_modularity, n_iters=500, seed=42)
        se_from_mod = eval_structural_entropy(adj, labels_mod)
        mod_from_mod = eval_modularity(adj, labels_mod)
        
        # Optimize Structural Entropy
        labels_se, _ = run_glass_jax_multistart(adj, k, two_dimensional_structural_entropy, n_iters=500, seed=42)
        se_from_se = eval_structural_entropy(adj, labels_se)
        mod_from_se = eval_modularity(adj, labels_se)
        
        print(f"{k:<4} | {'Glass-Mod':<20} | {se_from_mod:<15.4f} | {mod_from_mod:<20.4f}")
        print(f"{k:<4} | {'Glass-SE':<20} | {se_from_se:<15.4f} | {mod_from_se:<20.4f}")
        print("-" * 65)

if __name__ == "__main__":
    karate_adj, _, _ = get_karate_club()
    run_k_sweep(karate_adj, "Karate Club")
    
    sbm_adj, _, _ = get_sbm(150, 3, 0.3, 0.05, 42)
    run_k_sweep(sbm_adj, "SBM Clean (True K=3)")
