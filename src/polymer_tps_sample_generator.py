import numpy as np

from multiprocessing import Process, Pipe
import time
import os
from tqdm import tqdm

from tps import generate_path_to_state_brownian, h_A_B
from potential import Polymer

import json


class UnbiasedPolymer(Polymer):
    """Polymer that propagates on the unbiased force.

    tps.py propagates with potential.evaluate_force, but Polymer.evaluate_force is the biased force and expects the
    bias center appended to the coordinates. TPS here is unbiased, so evaluate_force is routed to the unbiased force.
    """

    def evaluate_force(self, coords):
        return self.evaluate_force_unbiased(coords)


def generate_paths(child_conn, worker_id, out_dir, n_samples, n_out, trajectory_out_frequency, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, initial_path, order_parameter_kwargs = {}, beta=1.0, max_length=10000, mute_warn=True):

    counter = 1
    samples_written = 0

    path = initial_path.copy()

    filename = os.path.join(out_dir, f"{worker_id}.json")

    with open(filename, "a") as out_file:
        while samples_written < n_samples:

            shooting_point = path[np.random.choice(len(path))]

            forward_traj, steps_forward = generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, trajectory_out_frequency=trajectory_out_frequency, beta=beta, max_steps=max_length, return_steps_needed=True, mute_warn=mute_warn, order_parameter_kwargs=order_parameter_kwargs)
            backward_traj = generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, trajectory_out_frequency=trajectory_out_frequency, beta=beta, max_steps=max_length - steps_forward, mute_warn=mute_warn, order_parameter_kwargs=order_parameter_kwargs)

            fw_in_A = h_A_B("A", forward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)
            fw_in_B = h_A_B("B", forward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)

            bw_in_A = h_A_B("A", backward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)
            bw_in_B = h_A_B("B", backward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)

            if (fw_in_A and bw_in_B) or (fw_in_B and bw_in_A):
                old_length = len(path)
                new_length = len(backward_traj) + len(forward_traj) - 1
                if np.random.random() < old_length/new_length:
                    path = np.vstack([backward_traj[::-1], forward_traj[1:]])

            if counter == n_out:
                write_result(path, out_file)
                samples_written += 1
                child_conn.send(worker_id)
                counter = 1
            else:
                counter += 1

    child_conn.close() # Signals the parent that this worker is done


def write_result(result, out_file):
    out_file.write(f"{json.dumps(result.tolist())}\n")
    out_file.flush()

def main():
    N_MONOMERS = 7
    DIMENSIONS = 2

    potential = UnbiasedPolymer(N_MONOMERS, DIMENSIONS)

    Q_A = 1.05
    Q_B = 1.2
    TIMESTEP = 1e-4
    DIFFUSION_COEFF = 1.0
    BETA = 1/0.1
    MAX_LENGTH = 1_000_000

    N_PROCESSES = 30

    N_SAMPLES = 30_000 # Per worker, so N_SAMPLES * N_PROCESSES paths in total
    N_OUT = 15
    TRAJECTORY_OUT_FREQUENCY = 10

    INITIAL_PATH_FILE = "samples/polymer_initial_paths.json"

    out_dir = "polymer_tps_normal"

    if os.path.isfile(out_dir):
        print(f"ERROR: {out_dir} exists and is a file. Aborting.")
        return
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        print(f"Directory {out_dir} exists and is not empty.")
        print("Aborting.")
        return
    os.makedirs(out_dir, exist_ok=True)

    initial_paths = []
    with open(INITIAL_PATH_FILE) as file:
        for line in file.read().splitlines():
            initial_paths.append(np.array(json.loads(line)))

    processes = []
    for worker_id in range(1, N_PROCESSES + 1):
        parent_conn, child_conn = Pipe()
        process = Process(target=generate_paths, args=(child_conn, worker_id, out_dir, N_SAMPLES, N_OUT, TRAJECTORY_OUT_FREQUENCY, Polymer.radius_of_gyration_2D, Q_A, Q_B, potential, TIMESTEP, DIFFUSION_COEFF, initial_paths[worker_id-1], {}, BETA, MAX_LENGTH))
        
        processes.append((process, parent_conn))
        process.start()
        child_conn.close() # Only the worker keeps its end open, so we see EOF once it exits
    
    pbar = tqdm(total=N_SAMPLES * N_PROCESSES, smoothing=0)

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
            except EOFError: # Worker wrote all of its samples and closed its end
                running.discard(parent_conn)
        time.sleep(0.1)

    pbar.close()

    for process, parent_conn in processes:
        process.join()


if __name__ == "__main__":
    main()
