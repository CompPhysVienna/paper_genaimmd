import numpy as np

from multiprocessing import Process, Pipe
import time
import os
from tqdm import tqdm

from tps import generate_path_to_state_brownian
from potential import WolfeQuapp


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


def wq_order_param_linear(coords):
    return coords[..., 0]/4 + coords[..., 1]


def wq_order_param_circle(coords, potential):
    prefactor = 100 # Determines the height of the function
    width = 10 # Determines the width of the peaks

    return prefactor * (-np.exp(-width*np.linalg.norm(coords - potential.A, axis=-1)**2) + np.exp(-width*np.linalg.norm(coords - potential.B, axis=-1)**2))

def q_A_B_from_R(A_B, R, potential):
    assert A_B.upper() in ["A", "B"], "A_B must be either A or B"
    peak = potential.A if A_B.upper() == "A" else potential.B
    return wq_order_param_circle(peak - np.array([R, 0.0]), potential)
    

def write_result(result, filename):
    with open(filename, "a") as file:
        batch = result[0]
        reached = result[1]
        for i in range(len(batch)):
            file.write(f"{batch[i][0]},{batch[i][1]},{reached[i]}\n")

def main():
    MAX = 2.3
    DX = 0.01
    N_TRAJECTORIES = 1000
    TARGET="B"
    
    H = 4.0
    potential = WolfeQuapp(h=H)
    
    Q_A = q_A_B_from_R("A", 0.5, potential)
    Q_B = q_A_B_from_R("B", 0.5, potential)
    TIMESTEP = 1e-2
    DIFFUSION_COEFF = 1.0
    
    N_PROCESSES = 23
    BATCH_SIZE = 25

    ORDER_PARAM = "circular"

    param_dict= {"max": MAX, "dx": DX, "ntrajectories" : N_TRAJECTORIES, "h": H, "qa": Q_A, "qb": Q_B, "target": TARGET, "timestep": TIMESTEP, "diffusioncoeff": DIFFUSION_COEFF, "orderparam": ORDER_PARAM}
    
    filename = "committor_estimates/wq"
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

    points_x_lim = (-MAX, MAX)
    points_y_lim = (-MAX, MAX)

    X, Y = np.meshgrid(np.arange(points_x_lim[0], points_x_lim[1] + DX, DX), np.arange(points_y_lim[0], points_y_lim[1] + DX, DX))
    points = np.vstack([X.flatten(), Y.flatten()]).T
    del X, Y

    processes = []
    for _ in range(N_PROCESSES):
        parent_conn, child_conn = Pipe()
        process = Process(target=propagate_points, args=(child_conn, N_TRAJECTORIES, TARGET, wq_order_param_circle, Q_A, Q_B, potential, TIMESTEP, DIFFUSION_COEFF, {"potential": potential}))
        
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
