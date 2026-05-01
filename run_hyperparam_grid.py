"""Run hyperparameter sensitivity grid."""
from pathlib import Path
from src.config_loader import load_config
from src.experiments.runner import run_experiment

configs = sorted(Path("configs/hyperparam").glob("*.yaml"))
print(f"Running {len(configs)} hyperparam configs...")

results = {}
for p in configs:
    config = load_config(p)
    name = config["experiment"]["name"]
    eps = config["q_learning"]["epsilon"]
    alpha = config["q_learning"]["alpha"]
    _, _, stat = run_experiment(config)
    _skip = {"comparisons", "_meta"}
    ql_key = [k for k in stat if k not in _skip and k != "CFR"][0]
    post = stat[ql_key]["post_mean"]
    results[name] = {"eps": eps, "alpha": alpha, "post": post}
    print(f"  {name}: eps={eps} alpha={alpha} post={post:+.4f}")

print()
print("=== Hyperparameter Sensitivity Grid ===")
print(f"{'':>12s}  alpha=0.01  alpha=0.1  alpha=0.3")
for e in [0.05, 0.15, 0.30]:
    row = f"eps={e:.2f}   "
    for a in [0.01, 0.1, 0.3]:
        match = [v for v in results.values()
                 if abs(v["eps"] - e) < 0.001 and abs(v["alpha"] - a) < 0.001]
        if match:
            row += f"  {match[0]['post']:+.4f}  "
        else:
            row += "    N/A    "
    print(row)
