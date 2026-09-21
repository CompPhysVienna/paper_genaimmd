import numpy as np

from multiprocessing import Process, Pipe
import time
import os
from tqdm import tqdm

from tps import generate_path_to_state_brownian
from potential import Polymer


def propagate_points(child_conn, n_trajectories, target, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, order_parameter_kwargs = {}, beta=1.0, max_steps=100000, mute_warn=False):
    child_conn.send("ready")
    while True:
        if not child_conn.poll():
            time.sleep(0.01)
            continue
        batch = child_conn.recv()
        reached = np.zeros(len(batch), dtype=float)
        for i, point in enumerate(batch):
            for _ in range(n_trajectories):
                reached[i] += generate_path_to_state_brownian(point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, target, order_parameter_kwargs, beta, max_steps, mute_warn)
        child_conn.send([batch, reached/n_trajectories])

    
def write_result(result, filename):
    with open(filename, "a") as file:
        batch = result[0]
        reached = result[1]
        for i in range(len(batch)):
            file.write(",".join(map(str, batch[i])) + f",{reached[i]}\n")

def main():
    N_TRAJECTORIES = 250
    TARGET = "B"
    
    potential = Polymer(7, 2)
    
    Q_A = 1.05
    Q_B = 1.2
    TIMESTEP = 1e-4
    DIFFUSION_COEFF = 1.0
    
    N_PROCESSES = 22
    BATCH_SIZE = 25
    BETA = 1/0.1

    param_dict= {"ntrajectories" : N_TRAJECTORIES, "beta": BETA, "qa": Q_A, "qb": Q_B, "target": TARGET, "timestep": TIMESTEP, "diffusioncoeff": DIFFUSION_COEFF}
    
    filename = "committor_estimates/polymer_gyration"
    for param_name, param in param_dict.items():
        filename += f"_{param_name}_{param}"
    filename += ".csv"

    if os.path.isdir(filename):
        print(f"ERROR: {filename} exists and is a directory. Aborting.")
    if os.path.isfile(filename):
        if input(f"File {filename} exists. Replace? [y/N]: ").lower() != "y":
            print("Aborting.")
            return
        os.remove(filename)

    points = np.loadtxt("samples/polymer_mcmc_beta_10_exchangerate_2_nout_1000_kbias_1500_nbiascenters_22_new.csv", delimiter=",")[::20, :-1]

    processes = []
    for _ in range(N_PROCESSES):
        parent_conn, child_conn = Pipe()
        process = Process(target=propagate_points, args=(child_conn, N_TRAJECTORIES, TARGET, Polymer.radius_of_gyration_2D, Q_A, Q_B, potential, TIMESTEP, DIFFUSION_COEFF, {}, BETA, 1000000))
        
        processes.append((process, parent_conn))
        process.start()

    n_batches_total = int(np.ceil(len(points)/BATCH_SIZE))
    batches_done = 0
    next_batch = 0
    
    pbar = tqdm(total=len(points), smoothing=0)

    while batches_done < n_batches_total:
        for process_index, state in enumerate(processes):
            process, parent_conn = state
            if parent_conn.poll():
                signal = parent_conn.recv()
                if isinstance(signal, list):
                    batches_done += 1
                    write_result(signal, filename)
                    if batches_done < n_batches_total:
                        pbar.update(BATCH_SIZE)
                    else:
                        pbar.update(len(points) - (n_batches_total - 1)*BATCH_SIZE)
                if next_batch < n_batches_total:
                    if next_batch != n_batches_total - 1:
                        batch = points[next_batch*BATCH_SIZE:(next_batch + 1)*BATCH_SIZE]
                    else:
                        batch = points[next_batch*BATCH_SIZE:]
                    parent_conn.send(batch)
                    next_batch += 1
                else:
                    process.terminate()
                    processes.pop(process_index)
        time.sleep(0.1)


if __name__ == "__main__":
    main()