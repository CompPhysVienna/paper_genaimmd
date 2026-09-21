import argparse
import os
import sys

import numpy as np

from analysis import common
from analysis.common import CLASS_KEYS, DATA_SETS, ERROR_STRIDE, REFERENCE, TRAJECTORY_OUT_FREQUENCY

def _count_classes(arguments):
    name, worker_id = arguments

    counts = np.zeros(len(CLASS_KEYS))
    for path in common.read_paths(common.data_set_file(name, worker_id)):
        common.class_histogram(path, out=counts)

    return counts


def class_counts(name=REFERENCE):
    results = common.parallel_map(_count_classes, [(name, i) for i in range(1, common.N_FILES + 1)],
                                  f"class counts ({name})")

    counts = np.sum(results, axis=0)
    out_file = common.output_file(name, "reference_class_counts.csv")
    np.savetxt(out_file, np.column_stack([CLASS_KEYS, counts]), delimiter=",")

    print(f"{out_file}: {counts.sum():,.0f} frames in {int((counts > 0).sum())} of {len(CLASS_KEYS)} classes")


def _measure_path_lengths(arguments):
    name, worker_id = arguments

    with open(common.data_set_file(name, worker_id)) as file:
        return [len(line.split("],")) *TRAJECTORY_OUT_FREQUENCY for line in file]


def path_lengths(names):
    for name in names:
        rows = common.parallel_map(_measure_path_lengths, [(name, i) for i in range(1, common.N_FILES + 1)],
                                   f"path lengths ({name})")

        lengths = common.stack(rows, f"{name}/path_lengths")
        out_file = os.path.join(DATA_SETS[name]["directory"], "path_lengths.csv")
        np.savetxt(out_file, lengths, delimiter=",")

        print(f"{out_file}: {lengths.shape} paths, mean {lengths.mean():.0f} steps")


def _reference_distribution():
    counts = np.loadtxt(common.output_file(REFERENCE, "reference_class_counts.csv"), delimiter=",")
    counts = counts[np.argsort(counts[:, 0])]

    return counts[:, 1]/counts[:, 1].sum()


def _error_curve(arguments):
    name, worker_id = arguments

    reference = _reference_distribution()
    weights = common.path_weights(name, worker_id)

    counts = np.zeros(len(CLASS_KEYS))
    errors = []

    for counter, path in enumerate(common.read_paths(common.data_set_file(name, worker_id)), start=1):
        weight = 1.0 if weights is None else weights[counter - 1]

        if weight > 0.0:
            common.class_histogram(path, weight=weight, out=counts)

        if counter % ERROR_STRIDE == 0:
            errors.append(np.abs(counts/counts.sum() - reference).sum())

    return errors


def errors(names):
    for name in names:
        rows = common.parallel_map(_error_curve, [(name, i) for i in range(1, common.N_FILES + 1)],
                                   f"error curves ({name})")

        all_errors = common.stack(rows, f"{name}/errors")
        out_file = common.output_file(name, "errors.csv")
        np.savetxt(out_file, all_errors, delimiter=",")

        print(f"{out_file}: {all_errors.shape}, final error {all_errors[:, -1].mean():.3f} "
              f"(min {all_errors[:, -1].min():.3f}, max {all_errors[:, -1].max():.3f})")

def _correlation_function(arguments):
    name, worker_id, lengths = arguments

    weights = common.path_weights(name, worker_id)
    if weights is not None:
        chosen = np.random.choice(len(lengths), len(lengths), replace=True, p=weights/np.sum(weights))
        lengths = lengths[chosen]

    return common.correlate(lengths, lengths)


def correlations(names):
    for name in names:
        lengths = np.loadtxt(os.path.join(DATA_SETS[name]["directory"], "path_lengths.csv"), delimiter=",")

        rows = common.parallel_map(_correlation_function,
                                   [(name, i + 1, lengths[i]) for i in range(len(lengths))],
                                   f"correlation functions ({name})")

        functions = common.stack(rows, f"{name}/correlation_functions")
        out_file = os.path.join(DATA_SETS[name]["directory"], "correlation_function_path_lengths.csv")
        np.savetxt(out_file, functions, delimiter=",")

        tau = 0.5 + np.sum(np.mean(functions, axis=0)[1:1001])
        print(f"{out_file}: {functions.shape}, tau_int over 1000 lags = {tau:.1f} samples")


STEPS = ("class_counts", "path_lengths", "errors", "correlations")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("steps", nargs="+", choices=STEPS + ("all",))
    parser.add_argument("--datasets", nargs="+", default=list(DATA_SETS), choices=list(DATA_SETS))
    parser.add_argument("--processes", type=int, default=common.N_PROCESSES)
    parser.add_argument("--fold", action="store_true",
                        help="mirror every configuration onto the half with a positive first internal angle "
                             "before classifying it, which takes the mirror symmetry out of p(x|TP). "
                             "class_counts and errors then write to <name>_folded.csv")
    arguments = parser.parse_args(argv)

    common.N_PROCESSES = arguments.processes
    common.FOLD_TO_POSITIVE_FIRST_ANGLE = arguments.fold
    steps = STEPS if "all" in arguments.steps else arguments.steps

    for step in steps:
        if step == "class_counts":
            class_counts()
        elif step == "path_lengths":
            path_lengths(arguments.datasets)
        elif step == "errors":
            errors([name for name in arguments.datasets if name != REFERENCE])
        elif step == "correlations":
            correlations([name for name in arguments.datasets if name != REFERENCE])


if __name__ == "__main__":
    main()
