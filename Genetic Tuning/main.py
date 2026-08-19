"""
main.py

Runs the full parameter-tuning pipeline of Section 4.3, "Parameter
Tuning: Genetic Algorithm", for real, using REPRESENTATIVE_SIZES and
FULL_GRID from tuning_pipeline.py. Each
selection scheme ("tournament" and "fitness_prop", Section 3.6) is run
completely independently.
"""

import pickle
import time

from tuning_pipeline import (
    REPRESENTATIVE_SIZES, FULL_GRID,
    tune_representative_sizes,
    generalization_check,
    retune_flagged,
)

SELECTION_SCHEMES = ("tournament", "fitness_prop")

# The tuning numbers reported in the thesis (Section 4.3, item 1): 30
# candidates, 6 instances, 4 repeats each, reduced budget of 1000
# generations.
TUNING_SETTINGS = dict(
    num_candidates=30,
    num_instances=6,
    num_repeats=4,
    max_gens=1000,
)
# The generalization check's settings (Section 4.3, item 3): 6 instances,
# 4 repeats, threshold 0.8 success rate -- matches "all sizes showed at
# least 0.8 success rate, finalizing the tuning."
GENERALIZATION_SETTINGS = dict(
    num_instances=6,
    num_repeats=4,
    max_gens=1000,
    success_rate_threshold=0.8,
)
RETUNING_SETTINGS = TUNING_SETTINGS

def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  [checkpoint saved: {path}]")


def tune_quantitative_params(selection_scheme):
    """Tunes pop_size / elite_size / cross_rate / mut_rate for each
    representative size (Section 4.3, item 1) and checkpoints the
    result.
    """
    print(f"=== Tuning: tuning {len(REPRESENTATIVE_SIZES)} representative sizes "
          f"for '{selection_scheme}' ===")
    tuned_buckets = tune_representative_sizes(
        selection_scheme,
        representative_sizes=REPRESENTATIVE_SIZES,
        **TUNING_SETTINGS,
    )
    save_checkpoint(tuned_buckets, f"tuned_buckets_{selection_scheme}_tuning.pkl")
    print()
    return tuned_buckets


def check_generalization(tuned_buckets, selection_scheme):
    """Tests each bucket's tuned params across the full grid and
    reports which sizes fall short of the success-rate threshold
    (Section 4.3, item 3), checkpointing the report.
    """
    print(f"=== Generalizing: checking generalization across all {len(FULL_GRID)} sizes ===")
    report, flagged_sizes = generalization_check(
        tuned_buckets, selection_scheme,
        grid=FULL_GRID,
        **GENERALIZATION_SETTINGS,
    )

    for (n, m), result in report.items():
        note = "  <-- FLAGGED" if (n, m) in flagged_sizes else ""
        print(f"  (n={n}, m={m}): success_rate={result['success_rate']:.2f} "
              f"(used bucket {result['used_bucket']}){note}")
    print(f"-> {len(flagged_sizes)} of {len(FULL_GRID)} sizes flagged for retuning")

    save_checkpoint({"report": report, "flagged": flagged_sizes},
                     f"generalization_report_{selection_scheme}.pkl")
    print()
    return flagged_sizes


def retune_failed_sizes(flagged_sizes, selection_scheme):
    """Retunes just the sizes that check_generalization flagged below
    the success-rate threshold -- not exercised for the thesis's final
    parameters, since every size already met the threshold (Section
    4.3).
    """
    print(f"=== Retuning: retuning {len(flagged_sizes)} flagged sizes ===")
    retuned = retune_flagged(flagged_sizes, selection_scheme, **RETUNING_SETTINGS)
    print()
    return retuned

def run_pipeline_for_scheme(selection_scheme):
    """Runs the full Section 4.3 tuning pipeline for one selection
    scheme -- tune, check generalization, retune anything flagged and returns its tuned
    buckets.
    """
    start_time = time.time()
    print(f"\n########## SCHEME: {selection_scheme} ##########")

    tuned_buckets = tune_quantitative_params(selection_scheme)

    flagged_sizes = check_generalization(tuned_buckets, selection_scheme)
    if flagged_sizes:
        tuned_buckets.update(retune_failed_sizes(flagged_sizes, selection_scheme))
        save_checkpoint(tuned_buckets, f"tuned_buckets_{selection_scheme}_final.pkl")
    else:
        print("=== Retuning: nothing flagged, skipping ===\n")

    elapsed_hours = (time.time() - start_time) / 3600
    print(f"Scheme '{selection_scheme}' finished in {elapsed_hours:.2f} hours\n")
    return tuned_buckets


def main():
    print(f"Representative sizes ({len(REPRESENTATIVE_SIZES)}): {REPRESENTATIVE_SIZES}")
    print(f"Full grid ({len(FULL_GRID)} sizes): {FULL_GRID}")

    all_tuned_params = {}
    for selection_scheme in SELECTION_SCHEMES:
        all_tuned_params[selection_scheme] = run_pipeline_for_scheme(selection_scheme)

    save_checkpoint(all_tuned_params, "tuned_buckets_all_schemes_final.pkl")
    print("All schemes done.")


if __name__ == "__main__":
    main()