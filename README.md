# bep-fair-division
Code accompanying a Bachelor's thesis on finding EFX (envy-free up to any good) allocations in fair division problems. Implements and empirically compares five algorithmic approaches — four ILP formulations, a Ratio-Greedy heuristic, Round-Robin, Simulated Annealing, and a Genetic Algorithm — across synthetic problem instances.

- **`Experimental Run/`** — Core algorithm implementations, objective functions, and instance generators (`algorithms.py`), the experiment runner that evaluates all algorithms across problem sizes (`experiment.py`), and the entry point for running experiments on a compute cluster (`run_experiments.py`).
- **`Genetic Tuning/`** — Parameter tuning pipeline for the Genetic Algorithm: the EA engine (`ea_core.py`), the tuning/validation logic (`tuning_pipeline.py`), and the script that runs the full tuning pipeline (`main.py`).
- **`Visualizations/`** — Notebook (`analyse_results.ipynb`) that turns experiment result CSVs into the tables and figures used in the thesis.
