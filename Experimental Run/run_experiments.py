"""
run_experiments.py

Entry point that runs Experiment 1 and/or Experiment 2 (Section 4,
"Experimental Methodology") on Snellius, sharing one process pool
across whichever experiments are given on the command line.

Usage:
    python3 run_experiments.py exp1
    python3 run_experiments.py exp2
    python3 run_experiments.py exp1 exp2      # both, one after another

Experiment 2's GA runs use experiment.GA_TUNED_PARAMS -- the tuned
parameters from Section 4.3, "Parameter Tuning: Genetic Algorithm".
Experiment 2 also needs Gurobi for the 4 ILP formulations of Section
3.2 (GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID must be set in the
environment before running).
"""

import os
import sys
from concurrent.futures import ProcessPoolExecutor

import experiment as E

# One worker process per allocated core (Section 4: "128 cores each ...
# embarrassingly parallel ... via a process pool").
N_WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))


# Runs experiment.run_experiment for each requested experiment ("exp1",
# "exp2", or both), writing results_experiment1.csv / results_experiment2.csv.
def main():
    experiments_to_run = sys.argv[1:] or ["exp1", "exp2"]
    valid = {"exp1", "exp2"}
    for name in experiments_to_run:
        if name not in valid:
            raise ValueError(f"Unknown experiment {name!r}; choose from {valid}")

    print(f"Worker processes: {N_WORKERS} "
          f"(from SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')})")

    with ProcessPoolExecutor(max_workers=N_WORKERS, initializer=E._init_worker) as executor:
        if "exp1" in experiments_to_run:
            E.run_experiment(E.EXPERIMENT_1, "results_experiment1.csv", executor=executor,
                              base_seed=0, progress_label="Experiment 1")
        if "exp2" in experiments_to_run:
            E.run_experiment(E.EXPERIMENT_2, "results_experiment2.csv", executor=executor,
                              base_seed=1_000_000, ga_tuned_params=E.GA_TUNED_PARAMS,
                              progress_label="Experiment 2")


if __name__ == "__main__":
    main()