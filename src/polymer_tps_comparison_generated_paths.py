import numpy as np
import torch

from generator import GeneratorBuilder
from potential import Polymer
from aimmd import AIMMDNetwork, AIMMDCommittorTrainer
from tps import generate_path_to_state_brownian

from transform_layers import PolymerTransformLayer, PolymerNormLayer

from multiprocessing import Process, Pipe
import time
import json
import os

from tqdm import tqdm


TRAINER_FILE = "trainers/polymer_adapted.pt"
OUT_DIR = "polymer_tps_generated"

N_MONOMERS = 7
DIMENSIONS = 2
K_BIAS = 2.5

Q_A = 1.05
Q_B = 1.2
TIMESTEP = 1e-4
DIFFUSION_COEFF = 1.0
TRAJECTORY_OUT_FREQUENCY = 10
BETA = 10.0
MAX_STEPS = 1_000_000

N_WORKERS = 30 # One worker per pre-generated shooting point file, i.e. resampled_sps_1.csv ... resampled_sps_N_WORKERS.csv

# The committor network is only evaluated for the path weights, the propagation itself runs on numpy. Keeping the
# network on the cpu avoids one cuda context per worker, which would not fit on the gpu.
DEVICE = "cpu"


def fold_to_positive_first_angle(coords):
    """Mirrors every configuration whose first internal angle is negative into the half with a positive one.

    The generator only proposes configurations with a positive first internal angle, the shooting points are
    mirrored with probability 1/2 afterwards. A frame with a negative first angle can therefore only have been
    proposed as the mirror image of its partner, and its density under rho_SP is the one of that partner.

    Parameters
    ----------
    coords : numpy.ndarray
        Cartesian coordinates of shape (n_configurations, n_monomers*dimensions)

    Returns
    -------
    numpy.ndarray
        Copy of the coordinates with the configurations of negative first internal angle mirrored
    """
    internal, _ = Polymer.cartesian_to_internal(coords)
    negative = internal[:, N_MONOMERS - 1] < 0

    folded = coords.copy()
    if negative.any():
        folded[negative] = Polymer.mirror(folded[negative])

    return folded


class UnbiasedPolymer(Polymer):
    """Polymer that propagates on the unbiased force.

    tps.py propagates with potential.evaluate_force, but Polymer.evaluate_force is the biased force and expects the
    bias center appended to the coordinates. The shots from the pre-generated shooting points are unbiased, so
    evaluate_force is routed to the unbiased force. The bias potential is still evaluated for the path weights.
    """

    def evaluate_force(self, coords):
        return self.evaluate_force_unbiased(coords)


def build_potential():
    """Builds the biased Polymer potential whose collective variable is the trained aimmd committor network.

    Every worker builds its own potential, so that no torch objects have to be shared between processes.

    Returns
    -------
    UnbiasedPolymer
        The Polymer potential with the committor network as its collective variable
    """
    trainer = torch.load(TRAINER_FILE, map_location=DEVICE, weights_only=False)
    aimmd_network = trainer.aimmd_network

    return UnbiasedPolymer(N_MONOMERS, dimensions=DIMENSIONS, cv_function=aimmd_network.cv_function,
                           cv_gradient_function=aimmd_network.cv_gradient_function, k_bias=K_BIAS)


def generate_paths(child_conn, worker_id, seed):
    """Shoots two trajectories towards state B from every pre-generated shooting point of this worker and writes the
    resulting transition paths and their weights to the files belonging to this worker.

    Parameters
    ----------
    child_conn : multiprocessing.connection.Connection
        The worker's end of the pipe to the parent. One worker_id is sent per written path, the end is closed once
        this worker is done
    worker_id : int
        Id of this worker, selects the shooting point file that is worked on and the files that are written
    seed : int
        Seed for the numpy rng of this worker. Workers would otherwise inherit the rng state of the parent
    """
    np.random.seed(seed)
    torch.set_num_threads(1)

    potential = build_potential()

    resampled_sps = np.loadtxt(os.path.join(OUT_DIR, f"resampled_sps_{worker_id}.csv"), delimiter=',')

    path_weights = []
    with open(os.path.join(OUT_DIR, f"{worker_id}.json"), 'w') as file:
        shooting_point_idx = 0
        while shooting_point_idx < len(resampled_sps):
            sp = resampled_sps[shooting_point_idx]
            fw_in_B, fw_path = generate_path_to_state_brownian(shooting_point=sp, order_parameter=Polymer.radius_of_gyration_2D, q_A=Q_A, q_B=Q_B, potential=potential, target='B', timestep=TIMESTEP, diffusion_coeff=DIFFUSION_COEFF, beta=BETA, trajectory_out_frequency=TRAJECTORY_OUT_FREQUENCY, max_steps=MAX_STEPS)
            bw_in_B, bw_path = generate_path_to_state_brownian(shooting_point=sp, order_parameter=Polymer.radius_of_gyration_2D, q_A=Q_A, q_B=Q_B, potential=potential, target='B', timestep=TIMESTEP, diffusion_coeff=DIFFUSION_COEFF, beta=BETA, trajectory_out_frequency=TRAJECTORY_OUT_FREQUENCY, max_steps=MAX_STEPS)

            reactive = fw_in_B ^ bw_in_B

            if reactive:
                path = np.vstack([bw_path[::-1], fw_path[1:]])

                folded = fold_to_positive_first_angle(path)
                single_points = np.exp(-BETA*potential.evaluate_bias_potential(folded, np.zeros(len(folded))))

                path_weight = 1/np.sum(single_points)
            elif shooting_point_idx == 0:
                continue
            else:
                path_weight = 0.0

            path_weights.append(path_weight)

            file.write(json.dumps(path.tolist()))
            file.write('\n')

            shooting_point_idx += 1
            child_conn.send(worker_id)

    np.savetxt(os.path.join(OUT_DIR, f"path_weights_{worker_id}.csv"), np.array(path_weights), delimiter=",")

    child_conn.close() # Signals the parent that this worker is done


def count_shooting_points(worker_id):
    """Counts the shooting points in the file belonging to a worker, one shooting point per line.

    Parameters
    ----------
    worker_id : int
        Id of the worker whose shooting point file is counted

    Returns
    -------
    int
        Number of shooting points of that worker
    """
    with open(os.path.join(OUT_DIR, f"resampled_sps_{worker_id}.csv")) as sps_file:
        return sum(1 for _ in sps_file)


def main():
    worker_ids = list(range(1, N_WORKERS + 1))

    missing = [worker_id for worker_id in worker_ids if not os.path.isfile(os.path.join(OUT_DIR, f"resampled_sps_{worker_id}.csv"))]
    if missing:
        print(f"ERROR: Missing shooting point files in {OUT_DIR} for worker(s) {missing}.")
        print("Run polymer_tps_comparison.ipynb first or lower N_WORKERS. Aborting.")
        return

    n_shooting_points = {worker_id: count_shooting_points(worker_id) for worker_id in worker_ids}

    seeds = np.random.SeedSequence().generate_state(N_WORKERS)

    processes = []
    for worker_id in worker_ids:
        parent_conn, child_conn = Pipe()
        process = Process(target=generate_paths, args=(child_conn, worker_id, int(seeds[worker_id - 1])))

        processes.append((process, parent_conn))
        process.start()
        child_conn.close() # Only the worker keeps its end open, so we see EOF once it exits

    pbar = tqdm(total=sum(n_shooting_points.values()), smoothing=0)

    running = {parent_conn for _, parent_conn in processes}
    while running:
        for parent_conn in list(running):
            try:
                while parent_conn.poll():
                    signal = parent_conn.recv()
                    if isinstance(signal, int):
                        pbar.update()
                    else:
                        raise ValueError(f"Got illegal signal of type {type(signal)}: {signal}")
            except EOFError: # Worker wrote all of its paths and closed its end
                running.discard(parent_conn)
        time.sleep(0.1)

    pbar.close()

    for process, parent_conn in processes:
        process.join()


if __name__ == "__main__":
    main()
