"""
tuning_pipeline.py

Implements the Genetic Algorithm parameter tuning described in Section
4.3, "Parameter Tuning: Genetic Algorithm". Population size, elite
size, crossover rate and mutation rate are tuned once per selection
scheme ("tournament" and "fitness_prop", Section 3.6), since the
thesis tunes the two schemes completely independently.

Section 4.3 describes tuning in three steps: (1) tune on five
representative (n, m) sizes by random search, testing each candidate
parameter vector on several random instances with several repeats
each, since a single EA run is too noisy to trust alone; (2) the
resulting winning parameters turned out identical across all five
representative sizes, so they were generalized to the whole instance
grid; (3) check whether those winning parameters remained widely
applicable across the full grid -- every size reached at least the 0.8
success-rate threshold, finalizing the tuning.

Below, random_search_tune / tune_representative_sizes implement step
1; generalization_check implements step 3, and reports which sizes (if
any) fall short of the success-rate threshold; retune_flagged repeats
step 1's random search restricted to those sizes, if any are ever
flagged (none were, for the thesis's final parameters).

Glossary (these words show up everywhere below):
  params              - one EA parameter vector: pop_size, elite_size,
                         cross_rate, mut_rate
  bucket              - one representative (n, m) size with its own tuned
                         params, stored as:
                         {(n, m): {"params": ..., "metrics": ..., "history": ...}}
  success_rate        - fraction of runs that found a violation-free (EFX)
                         allocation within the generation budget
  mean_best_fitness   - average, across runs, of each run's final violation
                         count (lower is better; 0 means solved)
  avg_gens_to_success - average number of generations used, counting only
                         the runs that actually succeeded
"""

import random
import statistics
import csv

from ea_core import generate_utilities, run_ea



# Same n values and m/n ratios as Experiment 2's grid (Section 4.2), since
# the GA is tuned on exactly the sizes it will later be evaluated on.
N_VALUES = [4, 6, 8, 12, 16]
M_TO_N_RATIOS = [1, 2, 3.5, 5, 10, 20]


def build_grid(n_values=N_VALUES, ratios=M_TO_N_RATIOS):
    """Expands n_values x ratios into the list of (n, m) sizes to test,
    rounding m = n*ratio to the nearest integer -- the same grid
    Experiment 2 uses (Section 4.2).
    """
    sizes = set()
    for n in n_values:
        for ratio in ratios:
            m = max(1, round(n * ratio))
            sizes.add((n, m))
    return sorted(sizes)


FULL_GRID = build_grid()

# The five representative sizes tuning is done on (Section 4.3, item 1):
# (4,4), (8,16), (12,42), (16,160), (16,320).
REPRESENTATIVE_SIZES = [
    (4, 4),
    (8, 16),
    (12, 42),
    (16, 160),
    (16, 320),
]



# The random-search ranges tuning draws candidates from (Section 4.3,
# item 1): pop_size in {20,40,80,150}, elite_fraction/cross_rate/mut_rate
# uniform on the given intervals.
PARAM_SEARCH_RANGES = {
    "pop_size": [20, 40, 80, 150],
    "elite_fraction": (0.02, 0.20),
    "cross_rate": (0.4, 0.95),
    "mut_rate": (0.001, 0.10),
}


def sample_random_params(rng, search_ranges):
    """Draws one random EA parameter vector from search_ranges, the
    random-search step of Section 4.3's tuning: pop_size is chosen
    from a fixed list, elite_fraction / cross_rate / mut_rate are
    drawn uniformly from their ranges, and elite_size is derived from
    pop_size * elite_fraction since elite size depends on population
    size.
    """
    pop_size = rng.choice(search_ranges["pop_size"])
    elite_fraction = rng.uniform(*search_ranges["elite_fraction"])

    elite_size = round(pop_size * elite_fraction)
    elite_size = max(1, min(pop_size - 1, elite_size))

    return {
        "pop_size": pop_size,
        "elite_size": elite_size,
        "cross_rate": rng.uniform(*search_ranges["cross_rate"]),
        "mut_rate": rng.uniform(*search_ranges["mut_rate"]),
    }


def evaluate_params(n, m, params, selection_scheme,
                     num_instances, num_repeats, max_gens, base_seed=0):
    """Runs the EA `num_repeats` times on each of `num_instances` random
    utility matrices with a fixed `params` vector, and summarizes how
    well it did: success_rate, mean_best_fitness, avg_gens_to_success
    (Section 4.3's tuning methodology).

    n, m             - problem size: n agents, m items
    params           - the EA parameter vector being tested
    selection_scheme - "tournament" or "fitness_prop" (Section 3.6)
    num_instances    - how many different random utility matrices to try
    num_repeats      - how many EA runs per utility matrix (the EA is
                        stochastic, so one run per instance isn't enough
                        to trust)
    max_gens         - generation budget for each individual run
    base_seed        - starting random seed, so results are reproducible
    """
    num_successes = 0
    final_best_scores = []
    gens_used_on_success = []

    seed = base_seed
    for _ in range(num_instances):
        random.seed(seed); seed += 1
        utility_matrix = generate_utilities(n, m)

        for _ in range(num_repeats):
            random.seed(seed); seed += 1
            result = run_ea(n, m, params["pop_size"], params["elite_size"],
                             params["cross_rate"], params["mut_rate"],
                             selection_scheme, utility_matrix, max_gens)

            final_best_scores.append(result["best"])
            if result["success"]:
                num_successes += 1
                gens_used_on_success.append(result["num_gens"])

    num_runs = num_instances * num_repeats
    return {
        "success_rate": num_successes / num_runs,
        "mean_best_fitness": statistics.mean(final_best_scores),
        "avg_gens_to_success": (
            statistics.mean(gens_used_on_success) if gens_used_on_success else float("inf")
        ),
    }


def rank_key(metrics):
    """Turns a metrics dict into something sortable, so the best candidate
    is just the one with the largest rank_key(...). Higher is always
    better, matching the three performance measures Section 4.3 ranks
    candidates by:
      1st priority: higher success_rate
      2nd priority: lower mean_best_fitness (fewer violations)
      3rd priority: fewer generations needed to succeed
    """
    gens = metrics["avg_gens_to_success"]
    fewer_gens_is_better = -gens if gens != float("inf") else float("-inf")
    return (metrics["success_rate"], -metrics["mean_best_fitness"], fewer_gens_is_better)


def random_search_tune(n, m, selection_scheme,
                        search_ranges=None, num_candidates=25,
                        num_instances=5, num_repeats=3,
                        max_gens=2000, seed=0):
    """Random search over EA parameters for a single (n, m) size (Section
    4.3, item 1): try `num_candidates` random parameter vectors, score
    each with evaluate_params, and keep the best by rank_key. The
    thesis's own run used num_candidates=30, num_instances=6,
    num_repeats=4, max_gens=1000 (see main.py).

    n, m              - problem size: n agents, m items
    selection_scheme  - "tournament" or "fitness_prop" (Section 3.6)
    search_ranges     - where candidates are drawn from (defaults to
                         PARAM_SEARCH_RANGES)
    num_candidates    - how many random parameter vectors to try
    num_instances     - passed through to evaluate_params, per candidate
    num_repeats       - passed through to evaluate_params, per candidate
    max_gens          - generation budget per run, while searching
    seed              - controls which random candidates get drawn (a
                         different seed is used per candidate when calling
                         evaluate_params, so results stay reproducible)

    Returns {"params": best params found, "metrics": its metrics,
             "history": every (params, metrics) pair tried}.
    """
    search_ranges = search_ranges or PARAM_SEARCH_RANGES
    rng = random.Random(seed)

    best_params, best_metrics = None, None
    history = []

    for i in range(num_candidates):
        candidate_params = sample_random_params(rng, search_ranges)
        candidate_metrics = evaluate_params(
            n, m, candidate_params, selection_scheme,
            num_instances, num_repeats, max_gens,
            base_seed=seed * 10_000 + i,
        )
        history.append((candidate_params, candidate_metrics))

        if best_metrics is None or rank_key(candidate_metrics) > rank_key(best_metrics):
            best_params, best_metrics = candidate_params, candidate_metrics

    return {"params": best_params, "metrics": best_metrics, "history": history}


def tune_representative_sizes(selection_scheme,
                               representative_sizes=REPRESENTATIVE_SIZES,
                               **tuning_kwargs):
    """Runs random_search_tune once per representative size (Section 4.3,
    item 1's five representative sizes: (4,4), (8,16), (12,42),
    (16,160), (16,320)).

    tuning_kwargs are forwarded straight through to random_search_tune --
    e.g. num_candidates, num_instances, num_repeats, max_gens.

    Returns {(n, m): {"params": ..., "metrics": ..., "history": ...}}
    """
    buckets = {}
    for (n, m) in representative_sizes:
        result = random_search_tune(n, m, selection_scheme, **tuning_kwargs)
        buckets[(n, m)] = result

        metrics = result["metrics"]
        print(f"  bucket (n={n}, m={m}): "
              f"success_rate={metrics['success_rate']:.2f}, "
              f"mean_best_fitness={metrics['mean_best_fitness']:.2f}")
        print(f"    -> {result['params']}")

    return buckets



def find_nearest_bucket(n, m, bucket_sizes):
    """Given a target size (n, m), returns whichever tuned bucket size
    (Section 4.3's representative sizes) is closest. m and n are on
    very different scales (m up to 320, n only up to 16), so each
    difference is divided by its rough max value first, to keep the
    two roughly comparable.
    """
    def distance_to(bucket_size):
        bucket_n, bucket_m = bucket_size
        return abs(n - bucket_n) / 16.0 + abs(m - bucket_m) / 320.0

    return min(bucket_sizes, key=distance_to)


def generalization_check(tuned_buckets, selection_scheme,
                          grid=FULL_GRID,
                          num_instances=5, num_repeats=3,
                          max_gens=2000,
                          success_rate_threshold=0.8, base_seed=5000):
    """For every size in `grid`, borrows the nearest bucket's tuned
    params (find_nearest_bucket) and tests them there, the check
    described in Section 4.3, item 3: "checked whether the winning
    parameters remained widely applicable ... at least 0.8 success
    rate". Sizes whose success_rate drops below success_rate_threshold
    are added to `flagged`.

    Returns (report, flagged):
      report  - {(n, m): {"used_bucket": (n, m), **metrics}}
      flagged - list of (n, m) sizes below success_rate_threshold
    """
    report = {}
    flagged = []

    for (n, m) in grid:
        bucket = find_nearest_bucket(n, m, tuned_buckets.keys())
        params = tuned_buckets[bucket]["params"]

        metrics = evaluate_params(n, m, params, selection_scheme,
                                   num_instances, num_repeats, max_gens,
                                   base_seed=base_seed)
        report[(n, m)] = {"used_bucket": bucket, **metrics}

        if metrics["success_rate"] < success_rate_threshold:
            flagged.append((n, m))

    return report, flagged



def retune_flagged(flagged_sizes, selection_scheme, **tuning_kwargs):
    """Repeats the random-search tuning of random_search_tune, but only
    for the sizes in `flagged_sizes` (whichever sizes
    generalization_check found below success_rate_threshold).
    tuning_kwargs are forwarded the same way as in
    tune_representative_sizes. The thesis reports every size already
    met the threshold (Section 4.3), so this function was not actually
    needed to produce the final tuned parameters.
    """
    new_buckets = {}
    for (n, m) in flagged_sizes:
        result = random_search_tune(n, m, selection_scheme, **tuning_kwargs)
        new_buckets[(n, m)] = result
        print(f"  retuned (n={n}, m={m}): "
              f"success_rate={result['metrics']['success_rate']:.2f} -> {result['params']}")
    return new_buckets