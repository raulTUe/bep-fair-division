"""
experiment.py

Runs Experiment 1 and Experiment 2 (Section 4, "Experimental
Methodology"): builds every instance and (algorithm, instance) task
described there, runs them (in parallel on Snellius), and writes one
results_experiment{1,2}.csv row per run.

Instance-level fields (same across every algorithm run on that instance):
    instance_id, n, m, m_over_n, valuation_generator, p, phi, instance_seed,
    sigma_1, sigma_2, experiment_id

Run-level fields (one row per algorithm attempt):
    run_id, instance_id, algorithm_name, start_method, algorithm_seed,
    runtime_wall_seconds, runtime_cpu_seconds, objective_1_value,
    objective_2_value, objective_3_value, objective_4_value, is_efx,
    status, mip_gap, num_iterations, algorithm_config

Every (algorithm, instance) run is independent, so all of one
experiment's runs are submitted to a process pool at once (Section 4:
"128 cores each ... embarrassingly parallel ... via a process pool").
"""

import os
import csv
import time
import random as random_module
from concurrent.futures import ProcessPoolExecutor, as_completed

import algorithms as A


# Experiment 1 (Section 4.1, "Good-to-Agent Ratio and Warm Starts"): n in
# {8,15,25}, the same m/n ratio grid as the thesis, 50 instances per (n,m)
# setting, i.i.d. uniform valuations, and the 4 SA warm starts of Section
# 3.5's "Warm Starts".
EXPERIMENT_1 = dict(
    experiment_id="exp1",
    n_values=[8, 15, 25],
    m_over_n_ratios=[1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 8, 13, 20, 25, 50],
    trials_per_setting=50,
    valuation_generator="iid_uniform",
    algorithms=["SA-Random", "SA-Welfare", "SA-Greedy", "SA-RoundRobin"],
)

# Experiment 2 (Section 4.2, "Comparing Algorithms in Practice"): n in
# {4,6,8,12,16} plus the two extra sizes (3,6)/(5,5) from Bohm et al.,
# the resampling generator, and all twelve algorithms compared in the thesis.
EXPERIMENT_2 = dict(
    experiment_id="exp2",
    n_values=[4, 6, 8, 12, 16],
    m_over_n_ratios=[1, 2, 3.5, 5, 10, 20],
    extra_sizes=[(3, 6), (5, 5)],
    trials_per_setting=50,
    valuation_generator="resampling",
    algorithms=[
        "SA-Random", "SA-Welfare", "SA-Greedy", "SA-RoundRobin",
        "Greedy", "RoundRobin", "GA-fitness_prop", "GA-tournament",
        "ILP-1", "ILP-2", "ILP-3", "ILP-4",
    ],
)

# Headroom for the instance_id/run_id encoding below (build_instances), not
# an experiment parameter -- both experiments actually use trials_per_setting=50.
MAX_TRIALS_PER_SETTING = 1000
# Matches the thesis's Simulated Annealing safety cap max_steps=10^7
# (Section 3.5, "Design").
SA_MAX_STEPS = 10_000_000
# Full-budget generation cap for the GA's final runs, matching the
# 10,000-generation budget used once tuning is finished (Section 4.3).
GA_MAX_GENS = 10_000
# No time limit / no MIP gap: the ILPs are solved to proven optimality,
# matching the thesis's exact ILP formulations (Section 3.2).
ILP_TIME_LIMIT_SECONDS = None
ILP_MIP_GAP = None

# The winning GA parameters found by parameter tuning (Section 4.3):
# pop_size 150, elite size 25, cross_rate 0.4553, mut_rate 0.0155 -- the
# thesis reports these won at every representative size, generalized
# here to the whole grid via find_nearest_ga_bucket.
GA_TUNED_PARAMS = {
    "tournament": {
        (16, 320): {"pop_size": 150, "elite_size": 25, "cross_rate": 0.4553, "mut_rate": 0.0155},
    },
    "fitness_prop": {
        (16, 320): {"pop_size": 150, "elite_size": 25, "cross_rate": 0.4553, "mut_rate": 0.0155},
    },
}


# Expands n_values x ratios into the (n, m) size grid, rounding m = n*ratio
# to the nearest integer (Section 4.1).
def build_grid(n_values, ratios):
    sizes = set()
    for n in n_values:
        for ratio in ratios:
            m = max(1, round(n * ratio))
            sizes.add((n, m))
    return sorted(sizes)


def generate_instance(valuation_generator, n, m, seed, p=None, phi=None):
    """Generates one instance's utility matrix deterministically from
    `seed`, wrapping algorithms.py's two generators: i.i.d. uniform
    (Section 4.1) and the resampling model (Section 2.2). If p/phi are
    given explicitly (resampling only), uses them directly instead of
    sampling internally -- this is what lets build_instances assign
    (p, phi) itself via stratified_p_phi_combos below. Returns
    (v, p, phi); p/phi are None for iid_uniform.
    """
    random_module.seed(seed)
    if valuation_generator == "iid_uniform":
        v = A.generate_iid_uniform(n, m)
        return v, None, None
    elif valuation_generator == "resampling":
        if p is not None and phi is not None:
            v = A.generate_utility_matrix(n, m, p, phi)
            return v, p, phi
        v, p2, phi2 = A.generate_utilities_resampling(n, m)
        return v, p2, phi2
    else:
        raise ValueError(f"Unknown valuation_generator: {valuation_generator!r}")


def stratified_p_phi_combos(p_values=None, phi_values=None):
    """Every (p, phi) combination from Section 2.2's resampling-model
    grid, in a fixed order -- cycling through this list
    (trial_index % len(combos)) spreads Experiment 2's 50 trials per
    setting as evenly as possible across combinations (Section 4.2),
    matching Bohm et al.'s own grid.
    """
    p_values = p_values if p_values is not None else A.P_VALUES
    phi_values = phi_values if phi_values is not None else A.PHI_VALUES
    return [(p, phi) for p in p_values for phi in phi_values]


def build_instances(spec, base_seed):
    """Builds every instance for one experiment spec: one entry per
    (n, m, trial), covering both the ratio-grid sizes (Section 4.1's
    or Section 4.2's m/n grid) and any spec['extra_sizes'] (Section
    4.2's extra (3,6)/(5,5) sizes). Utility matrices are not generated
    here -- that happens inside the worker (_run_one_task below).

    For the resampling generator, (p, phi) is assigned via stratified
    round-robin over stratified_p_phi_combos() above, rather than
    sampled randomly, so every (n, m) setting gets each combination as
    even a share of trials as possible (Section 4.2). With the 6x7=42
    combinations of Section 2.2's grid and trials_per_setting=50, that
    means 8 combinations get 2 trials and 34 get 1 trial each.
    """
    settings = build_grid(spec["n_values"], spec["m_over_n_ratios"])
    for extra in spec.get("extra_sizes", []):
        if extra not in settings:
            settings.append(extra)
    settings = sorted(settings)

    if spec["trials_per_setting"] > MAX_TRIALS_PER_SETTING:
        raise ValueError(
            f"trials_per_setting ({spec['trials_per_setting']}) exceeds MAX_TRIALS_PER_SETTING "
            f"({MAX_TRIALS_PER_SETTING}) -- raise that constant first (and note doing so will "
            f"NOT disturb any already-computed run's IDs, since it only widens unused headroom)."
        )

    combos = stratified_p_phi_combos() if spec["valuation_generator"] == "resampling" else None

    instances = []
    for size_index, (n, m) in enumerate(settings):
        for trial in range(spec["trials_per_setting"]):
            p = phi = None
            if combos is not None:
                p, phi = combos[trial % len(combos)]
            instance_id = size_index * MAX_TRIALS_PER_SETTING + trial
            instances.append({
                "instance_id": instance_id,
                "experiment_id": spec["experiment_id"],
                "n": n, "m": m, "m_over_n": round(m / n, 4),
                "trial": trial,
                "valuation_generator": spec["valuation_generator"],
                "instance_seed": base_seed + instance_id,
                "p": p, "phi": phi,
            })
    return instances


def find_nearest_ga_bucket(n, m, bucket_sizes):
    """Same nearest-bucket distance metric as
    tuning_pipeline.find_nearest_bucket, used to pick which of
    GA_TUNED_PARAMS' representative-size buckets (Section 4.3) a given
    (n, m) in Experiment 2 should borrow its GA parameters from.
    """
    def distance_to(bucket_size):
        bn, bm = bucket_size
        return abs(n - bn) / 16.0 + abs(m - bm) / 320.0
    return min(bucket_sizes, key=distance_to)



# Each worker process builds its own Gurobi environment once (Gurobi
# environments can't be pickled across a process boundary) for the
# ILP-1..4 tasks below.
_WORKER_GUROBI_ENV = None


def _init_worker():
    global _WORKER_GUROBI_ENV
    if A.gp is not None:
        try:
            _WORKER_GUROBI_ENV = A.make_gurobi_env()
        except Exception:
            _WORKER_GUROBI_ENV = None


# Runs one (algorithm, instance) task: dispatches to whichever of the
# twelve algorithms (SA-* warm starts, Greedy, RoundRobin, GA-*, ILP-1..4)
# algorithm_name names -- every algorithm compared in Section 3 -- and
# records is_efx plus Obj_1..Obj_4 (evaluate_all_objectives) for the result row.
def _run_one_task(task):
    v, p, phi = generate_instance(task["valuation_generator"], task["n"], task["m"],
                                   task["instance_seed"], p=task.get("p"), phi=task.get("phi"))
    sigma_1, sigma_2 = A.explicit_map(v)
    n, m = task["n"], task["m"]

    random_module.seed(task["algorithm_seed"])

    algorithm_name = task["algorithm_name"]
    allocation = None
    status = "error"
    mip_gap = None
    num_iterations = None
    algorithm_config = ""
    start_method = None

    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    if algorithm_name.startswith("SA-"):
        start_method = algorithm_name[len("SA-"):]
        init_fn = {
            "Random": A.initialize_allocation,
            "Welfare": A.initialize_allocation_from_optimal_welfare,
            "Greedy": A.initialize_allocation_from_greedy,
            "RoundRobin": A.initialize_allocation_from_round_robin,
        }[start_method]
        bundles, owner, agent_utilities = init_fn(n, m, v)
        num_steps, final_bundles, _, _ = A.find_efx_allocation(
            n, m, v, bundles, owner, agent_utilities,
            max_steps=task["sa_max_steps"]
        )
        num_iterations = num_steps
        algorithm_config = f"max_steps={task['sa_max_steps']}"
        allocation = A.bundles_to_allocation(final_bundles)
        if A.is_efx(allocation, v, n, m):
            status = "success"
        elif num_steps >= task["sa_max_steps"]:
            status = "max_steps_reached"
        else:
            status = "cooling_exhausted"

    elif algorithm_name == "Greedy":
        alloc_raw, _ = A.find_efx_allocation_greedy(n, m, v)
        allocation = alloc_raw
        status = "completed"

    elif algorithm_name == "RoundRobin":
        bundles, owner, agent_utilities = A.round_robin_allocation(n, m, v)
        allocation = A.bundles_to_allocation(bundles)
        status = "completed"

    elif algorithm_name.startswith("GA-"):
        scheme = algorithm_name[len("GA-"):]
        params = task["ga_params"]
        result = A.run_ea(n, m, params["pop_size"], params["elite_size"],
                           params["cross_rate"], params["mut_rate"],
                           scheme, v, max_gens=task["ga_max_gens"])
        allocation = A.solution_to_allocation(n, result["best_solution"])
        num_iterations = result["num_gens"]
        status = "success" if result["success"] else "max_gens_reached"
        algorithm_config = (f"pop_size={params['pop_size']},elite_size={params['elite_size']},"
                             f"cross_rate={params['cross_rate']:.4f},mut_rate={params['mut_rate']:.4f},"
                             f"max_gens={task['ga_max_gens']}")

    elif algorithm_name.startswith("ILP-"):
        if _WORKER_GUROBI_ENV is None:
            raise RuntimeError(
                "ILP task dispatched but this worker has no Gurobi env "
                "(WLS credentials missing/invalid -- see make_gurobi_env)."
            )
        solver = {"ILP-1": A.solve_ilp_1, "ILP-2": A.solve_ilp_2,
                  "ILP-3": A.solve_ilp_3, "ILP-4": A.solve_ilp_4}[algorithm_name]
        M = [sum(row) for row in v]
        allocation, ilp_objval, status, mip_gap, _ = solver(
            n, m, v, M, _WORKER_GUROBI_ENV,
            time_limit=task["ilp_time_limit"], mip_gap=task["ilp_mip_gap"]
        )
        algorithm_config = f"time_limit={task['ilp_time_limit']},mip_gap_target={task['ilp_mip_gap']}"
        if allocation is not None:
            obj_index = int(algorithm_name[-1]) - 1
            standalone_val = A.evaluate_all_objectives(allocation, v)[obj_index]
            if abs(standalone_val - ilp_objval) > 1e-4:
                status = status + "|OBJECTIVE_MISMATCH"

    else:
        raise ValueError(f"Unknown algorithm_name: {algorithm_name!r}")

    wall_elapsed = time.perf_counter() - wall_start
    cpu_elapsed = time.process_time() - cpu_start

    if allocation is not None:
        obj1, obj2, obj3, obj4 = A.evaluate_all_objectives(allocation, v)
        efx = A.is_efx(allocation, v, n, m)
    else:
        obj1 = obj2 = obj3 = obj4 = None
        efx = None

    return {
        "run_id": task["run_id"],
        "instance_id": task["instance_id"],
        "experiment_id": task["experiment_id"],
        "n": n, "m": m, "m_over_n": task["m_over_n"],
        "valuation_generator": task["valuation_generator"],
        "p": p, "phi": phi,
        "instance_seed": task["instance_seed"],
        "sigma_1": sigma_1, "sigma_2": sigma_2,
        "algorithm_name": algorithm_name,
        "start_method": start_method,
        "algorithm_seed": task["algorithm_seed"],
        "runtime_wall_seconds": wall_elapsed,
        "runtime_cpu_seconds": cpu_elapsed,
        "objective_1_value": obj1, "objective_2_value": obj2,
        "objective_3_value": obj3, "objective_4_value": obj4,
        "is_efx": efx,
        "status": status,
        "mip_gap": mip_gap,
        "num_iterations": num_iterations,
        "algorithm_config": algorithm_config,
    }


# Cross-product of instances x algorithms into one task dict per
# (algorithm, instance) run, attaching the tuned GA parameters (via
# find_nearest_ga_bucket) where relevant.
def build_tasks(spec, instances, ga_tuned_params=None):
    tasks = []
    algorithms = spec["algorithms"]
    for instance in instances:
        for algo_index, algorithm_name in enumerate(algorithms):
            task = dict(instance)
            run_id = instance["instance_id"] * len(algorithms) + algo_index
            task["run_id"] = run_id
            task["algorithm_name"] = algorithm_name
            task["algorithm_seed"] = run_id + 500_000_000
            task["sa_max_steps"] = SA_MAX_STEPS
            task["ga_max_gens"] = GA_MAX_GENS
            task["ilp_time_limit"] = ILP_TIME_LIMIT_SECONDS
            task["ilp_mip_gap"] = ILP_MIP_GAP

            if algorithm_name.startswith("GA-"):
                if ga_tuned_params is None:
                    raise ValueError("Experiment includes GA algorithms but no ga_tuned_params were given.")
                scheme = algorithm_name[len("GA-"):]
                buckets = ga_tuned_params[scheme]
                nearest = find_nearest_ga_bucket(task["n"], task["m"], buckets.keys())
                task["ga_params"] = buckets[nearest]

            tasks.append(task)
    return tasks


# Column order for results_experiment{1,2}.csv, matching the run-level /
# instance-level schema documented in this file's module docstring.
RESULT_FIELDNAMES = [
    "run_id", "instance_id", "experiment_id", "n", "m", "m_over_n",
    "valuation_generator", "p", "phi", "instance_seed", "sigma_1", "sigma_2",
    "algorithm_name", "start_method", "algorithm_seed", "runtime_wall_seconds",
    "runtime_cpu_seconds", "objective_1_value", "objective_2_value",
    "objective_3_value", "objective_4_value", "is_efx", "status", "mip_gap",
    "num_iterations", "algorithm_config",
]


def load_completed_run_ids(out_csv):
    """Reads the run_id values already present in an existing results
    CSV, if any -- this is what makes an interrupted run resumable, so
    rerunning after an interruption only computes the runs that are
    still missing instead of starting the whole experiment over.
    """
    if not os.path.exists(out_csv):
        return set()
    completed = set()
    with open(out_csv, newline="") as f:
        for row in csv.DictReader(f):
            completed.add(int(row["run_id"]))
    return completed


# Top-level driver: builds instances/tasks for one experiment spec, skips
# already-completed run_ids, and appends each remaining run's result row to
# out_csv (via the process-pool executor, if one is given).
def run_experiment(spec, out_csv, executor=None, base_seed=0, ga_tuned_params=None,
                    progress_label=None):
    instances = build_instances(spec, base_seed)
    tasks = build_tasks(spec, instances, ga_tuned_params=ga_tuned_params)
    valid_run_ids = {t["run_id"] for t in tasks}

    completed_run_ids = load_completed_run_ids(out_csv)
    stale = completed_run_ids - valid_run_ids
    if stale:
        raise RuntimeError(
            f"{out_csv} already exists and contains {len(stale)} run_id(s) that don't match "
            f"the CURRENT experiment spec (the grid, algorithm list, or trial count must have "
            f"changed since that file was written). Refusing to resume from a mismatched file, "
            f"since silently merging would produce a corrupted dataset -- move or delete "
            f"{out_csv} first if you really want to start fresh with the new spec."
        )

    remaining_tasks = [t for t in tasks if t["run_id"] not in completed_run_ids]
    print(f"{spec['experiment_id']}: {len(instances)} instances x "
          f"{len(spec['algorithms'])} algorithms = {len(tasks)} total runs "
          f"({len(completed_run_ids)} already in {out_csv}, {len(remaining_tasks)} remaining)")

    if not remaining_tasks:
        print(f"Nothing left to run -- {out_csv} is already complete.")
        return

    file_is_new = len(completed_run_ids) == 0
    with open(out_csv, "w" if file_is_new else "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDNAMES)
        if file_is_new:
            writer.writeheader()
            f.flush()
            os.fsync(f.fileno())

        total = len(remaining_tasks)
        start = time.time()
        report_every = max(1, total // 20)
        done = 0

        if executor is not None:
            futures = [executor.submit(_run_one_task, t) for t in remaining_tasks]
            for future in as_completed(futures):
                row = future.result()
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                done += 1
                if progress_label and (done % report_every == 0 or done == total):
                    elapsed_min = (time.time() - start) / 60
                    print(f"  [{progress_label}] {done}/{total} runs done this session "
                          f"({elapsed_min:.1f} min elapsed)", flush=True)
        else:
            for t in remaining_tasks:
                row = _run_one_task(t)
                writer.writerow(row)
                f.flush()
                os.fsync(f.fileno())
                done += 1

    print(f"Done this session. {out_csv} now has {len(completed_run_ids) + done} "
          f"of {len(tasks)} total runs.")