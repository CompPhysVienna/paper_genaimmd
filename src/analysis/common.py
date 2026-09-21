import json
import multiprocessing
import os

import numpy as np
from tqdm import tqdm

N_PROCESSES = 30
N_FILES = 30
ERROR_STRIDE = 10
TRAJECTORY_OUT_FREQUENCY = 10

FOLD_TO_POSITIVE_FIRST_ANGLE = False

DATA_SETS = {
    "reference": {"directory": "polymer_tps_reference", "weighted": False},
    "normal": {"directory": "polymer_tps_normal", "weighted": False},
    "generated": {"directory": "polymer_tps_generated", "weighted": True},
}

REFERENCE = "reference"

def to_base(number, base=5):
    if not number:
        return 0

    digits = []
    while number:
        digits.append(str(number % base))
        number //= base

    return int("".join(list(reversed(digits))))


CLASS_KEYS = np.array([to_base(i) for i in range(5**5)])
_LOOKUP = np.full(CLASS_KEYS.max() + 1, -1, dtype=np.int32)
_LOOKUP[CLASS_KEYS] = np.arange(len(CLASS_KEYS))
_DIGIT_WEIGHTS = np.array([10000, 1000, 100, 10, 1])


def fold_to_positive_first_angle(frames):
    from potential import Polymer

    internal, _ = Polymer.cartesian_to_internal(np.asarray(frames))
    n_monomers = (internal.shape[-1] + 3)//2

    angles = internal[..., n_monomers - 1:]
    angles[angles[..., 0] < 0] *= -1

    return internal


def output_file(name, template):
    if FOLD_TO_POSITIVE_FIRST_ANGLE:
        stem, extension = os.path.splitext(template)
        template = f"{stem}_folded{extension}"

    return os.path.join(DATA_SETS[name]["directory"], template)


def class_histogram(frames, weight=1.0, out=None):
    from potential import Polymer

    frames = np.asarray(frames)
    if FOLD_TO_POSITIVE_FIRST_ANGLE:
        frames = fold_to_positive_first_angle(frames)

    classes = Polymer.get_polymer_class(frames)
    indices = _LOOKUP[classes @ _DIGIT_WEIGHTS]

    counts = np.bincount(indices, minlength=len(CLASS_KEYS))*weight
    if out is None:
        return counts

    out += counts
    return out


def read_paths(path_file):
    with open(path_file) as file:
        for line in file:
            yield np.array(json.loads(line))


def data_set_file(name, worker_id, template="{worker}.json"):
    return os.path.join(DATA_SETS[name]["directory"], template.format(worker=worker_id))


def path_weights(name, worker_id):
    if not DATA_SETS[name]["weighted"]:
        return None

    return np.loadtxt(data_set_file(name, worker_id, "path_weights_{worker}.csv"), delimiter=",")


def correlate(A, B, normalize=True):
    assert len(A) == len(B), f"Inputs have different lengths ({len(A)} != {len(B)})"
    M = len(A)

    com_A = A - np.mean(A)
    com_B = B - np.mean(B)

    C_AB = np.correlate(com_A, com_B, mode="full")[-M:] / np.arange(M, 0, -1)

    if normalize:
        C_AB /= np.std(com_A)*np.std(com_B)

    return C_AB


def parallel_map(function, arguments, description):
    context = multiprocessing.get_context("fork")

    with context.Pool(N_PROCESSES) as pool:
        return list(tqdm(pool.imap(function, arguments), total=len(arguments), desc=description, smoothing=0))


def stack(rows, name):
    lengths = {len(row) for row in rows}

    if len(lengths) > 1:
        shortest = min(lengths)
        print(f"WARNING: {name} has {sorted(lengths)} entries per file, trimming all to {shortest}")
        rows = [row[:shortest] for row in rows]

    return np.array(rows)
