import numpy as np

from multiprocessing import Process, Pipe
import time
import os
from tqdm import tqdm

from tps import generate_path_to_state_brownian, h_A_B
from potential import WolfeQuapp
from wq_estimate_committor import wq_order_param_circle, q_A_B_from_R
from util import graceful_keyboard_interrupt


@graceful_keyboard_interrupt()
def generate_paths(child_conn, n_out, trajectory_out_frequency, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, target, order_parameter_kwargs = {}, beta=1.0, max_steps=100000, mute_warn=False):

    counter = 1

    initial_path_x = np.linspace(potential.A[0], potential.B[0], 100)
    path = np.vstack([initial_path_x, initial_path_x]).T

    while True:
        
        shooting_point = path[np.random.choice(len(path))]

        forward_traj = generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, trajectory_out_frequency=trajectory_out_frequency, beta=beta, max_steps=max_steps, mute_warn=mute_warn, order_parameter_kwargs=order_parameter_kwargs)
        backward_traj = generate_path_to_state_brownian(shooting_point, order_parameter, q_A, q_B, potential, timestep, diffusion_coeff, trajectory_out_frequency=trajectory_out_frequency, beta=beta, max_steps=max_steps, mute_warn=mute_warn, order_parameter_kwargs=order_parameter_kwargs)

        fw_in_A = h_A_B("A", forward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)
        fw_in_B = h_A_B("B", forward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)

        bw_in_A = h_A_B("A", backward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)
        bw_in_B = h_A_B("B", backward_traj[-1], order_parameter, q_A, q_B, order_parameter_kwargs=order_parameter_kwargs)

        new_path = np.vstack([backward_traj[::-1], forward_traj[1:]])

        if (fw_in_A and bw_in_B) or (fw_in_B and bw_in_A):
            old_length = len(path)
            new_length = len(backward_traj) + len(forward_traj)
            if np.random.random() < old_length/new_length:
                path = new_path

        if counter == n_out:
            outcome_fw = 2*fw_in_A - 1 if target == "B" else 2*fw_in_B - 1
            outcome_bw = 2*bw_in_A - 1 if target == "B" else 2*bw_in_B - 1
            child_conn.send([*shooting_point, outcome_fw, outcome_bw])
            counter = 1
        else:
            counter += 1


def write_result(result, filename):
    with open(filename, "a") as file:
        file.write(",".join(map(str, result)) + "\n")

@graceful_keyboard_interrupt(message="\n\nAborting.")
def main():
    MAX = 2.3
    H = 4.0

    potential = WolfeQuapp(h=H)

    Q_A = q_A_B_from_R("A", 0.5, potential)
    Q_B = q_A_B_from_R("B", 0.5, potential)
    TIMESTEP = 1e-2
    DIFFUSION_COEFF = 1.0
    
    N_PROCESSES = 23
    
    N_SAMPLES = 25000
    N_OUT = 1000
    TRAJECTORY_OUT_FREQUENCY = 10

    ORDER_PARAM = "circular"

    TARGET = "B"

    param_dict= {"nsamples" : N_SAMPLES, "nout" : N_OUT, "trajectoryoutfrequency" : TRAJECTORY_OUT_FREQUENCY, "max": MAX, "h": H, "qa": Q_A, "qb": Q_B, "timestep": TIMESTEP, "diffusioncoeff": DIFFUSION_COEFF, "orderparam": ORDER_PARAM, "target": TARGET}
    
    filename = "samples/wq_sps"
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

    processes = []
    for _ in range(N_PROCESSES):
        parent_conn, child_conn = Pipe()
        process = Process(target=generate_paths, args=(child_conn, N_OUT, TRAJECTORY_OUT_FREQUENCY, wq_order_param_circle, Q_A, Q_B, potential, TIMESTEP, DIFFUSION_COEFF, TARGET, {"potential": potential}))
        
        processes.append((process, parent_conn))
        process.start()
    
    pbar = tqdm(total=N_SAMPLES, smoothing=0)
    samples_done = 0

    while samples_done < N_SAMPLES:
        for process, parent_conn in processes:
            if parent_conn.poll():
                signal = parent_conn.recv()
                if isinstance(signal, list):
                    samples_done += 1
                    write_result(signal, filename)
                    pbar.update()
                else:
                    raise ValueError(f"Got illegal signal of type {type(signal)}: {signal}")
        time.sleep(0.1)

    for process, parent_conn in processes:
        process.terminate()


if __name__ == "__main__":
    main()
