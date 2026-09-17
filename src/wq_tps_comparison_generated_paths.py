import numpy as np
import torch
import scipy as sc

from potential import WolfeQuapp, WolfeQuappBiased
from wq_estimate_committor import wq_order_param_circle, q_A_B_from_R
from tps import generate_path_to_state_brownian

from multiprocessing import Process, Pipe
import time
import json
import os

from tqdm import tqdm


OUT_DIR = "paper_wq_tps_generated"

h = 4.0

q_A = q_A_B_from_R("A", 0.5, WolfeQuapp(h=h))
q_B = q_A_B_from_R("B", 0.5, WolfeQuapp(h=h))

param_dict= {"max": 2.3, "dx": 0.01, "ntrajectories" : 1000, "h": h, "qa": q_A, "qb": q_B, "target": "B", "timestep": 1e-2, "diffusioncoeff": 1.0, "orderparam": "circular"}

filename = "committor_estimates/wq"
for param_name, param in param_dict.items():
    filename += f"_{param_name}_{param}"
filename += ".csv"

loaded_committor_estimates = np.loadtxt(filename, delimiter=",")

side_length = int(np.sqrt(len(loaded_committor_estimates)))
sort_args = np.argsort(loaded_committor_estimates[:, 0]*side_length + loaded_committor_estimates[:, 1])

committor_estimates = loaded_committor_estimates[sort_args]

X = committor_estimates[:, 0].reshape(side_length, side_length).T[0].flatten()
Y = committor_estimates[:, 1].reshape(side_length, side_length).T[:, 0].flatten()

committor_values = committor_estimates[:, 2].reshape(side_length, side_length)
committor_values = sc.ndimage.gaussian_filter(committor_values, 5)

interp = sc.interpolate.RegularGridInterpolator((X, Y), committor_values, bounds_error=False)

def cv_function(coords):
    coords = np.where(np.abs(coords) < 2.3, coords, np.sign(coords)*2.299)
    return interp(coords)

unwrapped_potential = WolfeQuappBiased(h=h, k_bias=300, cv_function=cv_function)
cv_gradient_function = unwrapped_potential.cv_gradient

class CVFunctionAutograd(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)

        coords = input.detach().cpu().numpy()
        cv_values = cv_function(coords)

        return torch.from_numpy(cv_values).to(input.device)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors

        coords = input.detach().cpu().numpy()
        grad = torch.from_numpy(cv_gradient_function(coords)).to(input.device)

        return grad*grad_output[..., None]
    
def cv_function_wrapper(coords):
    return CVFunctionAutograd.apply(coords)

potential = WolfeQuappBiased(h=h, k_bias=300, cv_function=cv_function_wrapper)


unbiased_potential = WolfeQuapp(h=h)

Q_A = q_A_B_from_R("A", 0.5, potential)
Q_B = q_A_B_from_R("B", 0.5, potential)
TIMESTEP = 1e-3
DIFFUSION_COEFF = 1.0
TRAJECTORY_OUT_FREQUENCY = 10
MAX_STEPS = 100000

N_WORKERS = 30 # One worker per pre-generated shooting point file, i.e. resampled_sps_1.csv ... resampled_sps_N_WORKERS.csv


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

    resampled_sps = np.loadtxt(os.path.join(OUT_DIR, f"resampled_sps_{worker_id}.csv"), delimiter=',')

    path_weights = []
    with open(os.path.join(OUT_DIR, f"{worker_id}.json"), 'w') as file:
        shooting_point_idx = 0
        while shooting_point_idx < len(resampled_sps):
            sp = resampled_sps[shooting_point_idx]
            fw_in_B, fw_path = generate_path_to_state_brownian(shooting_point=sp, order_parameter=wq_order_param_circle, q_A=Q_A, q_B=Q_B, potential=unbiased_potential, target='B', timestep=TIMESTEP, diffusion_coeff=DIFFUSION_COEFF, order_parameter_kwargs={"potential" : potential}, trajectory_out_frequency=TRAJECTORY_OUT_FREQUENCY, max_steps=MAX_STEPS)
            bw_in_B, bw_path = generate_path_to_state_brownian(shooting_point=sp, order_parameter=wq_order_param_circle, q_A=Q_A, q_B=Q_B, potential=unbiased_potential, target='B', timestep=TIMESTEP, diffusion_coeff=DIFFUSION_COEFF, order_parameter_kwargs={"potential" : potential}, trajectory_out_frequency=TRAJECTORY_OUT_FREQUENCY, max_steps=MAX_STEPS)

            reactive = fw_in_B ^ bw_in_B

            if reactive:
                path = np.vstack([bw_path[::-1], fw_path[1:]])
                single_points = np.exp(-unwrapped_potential.evaluate_bias_potential(path, 0.5*np.ones(len(path))))
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
        print("Run wq_tps_comparison.ipynb first or lower N_WORKERS. Aborting.")
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
