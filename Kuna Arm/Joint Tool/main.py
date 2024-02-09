import json
import sys
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import signal
from tqdm import tqdm
matplotlib.use('TkAgg')
signal.signal(signal.SIGINT, signal.SIG_DFL)

GRAVITY = 9.81

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def plot(td, pos):
    plt.figure()
    l23_t = np.radians(td["j2"][pos])
    j3_x = td["l23"]["len"] * np.cos(l23_t)
    j3_y = td["l23"]["len"] * np.sin(l23_t)
    plt.plot([td["j2"]["spg_start_x"], td["j2"]["spg_end_d"] * np.cos(l23_t)],
             [td["j2"]["spg_start_y"], td["j2"]["spg_end_d"] * np.sin(l23_t)], color='gray', marker='o')
    plt.plot([0, j3_x], [0, j3_y], color='cornflowerblue', marker='o')
    l34_t = np.radians(td["j3"][pos]) + l23_t
    j5_x = td["l34"]["len"] * np.cos(l34_t) + j3_x
    j5_y = td["l34"]["len"] * np.sin(l34_t) + j3_y
    plt.plot([td["j3"]["spg_start_x"] + j3_x, td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)], 
             [td["j3"]["spg_start_y"] + j3_y, td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t)], color='gray', marker='o')
    plt.plot([j3_x, j3_x - td["l34"]["offset"] * np.sin(l34_t)],
             [j3_y, j3_y + td["l34"]["offset"] * np.cos(l34_t)], color='royalblue', marker='o')
    plt.plot([j3_x - td["l34"]["offset"] * np.sin(l34_t), j5_x],
             [j3_y + td["l34"]["offset"] * np.cos(l34_t), j5_y], color='royalblue', marker='o')
    l4G_t = np.radians(td["j5"][pos]) + l34_t
    g_x = td["l4G"]["len"] * np.cos(l4G_t) + j5_x
    g_y = td["l4G"]["len"] * np.sin(l4G_t) + j5_y
    plt.plot([j5_x, g_x], [j5_y, g_y], color='blue', marker='o')
    lG_t = l4G_t
    e_x = td["lG"]["len"] * np.cos(lG_t) + g_x
    e_y = td["lG"]["len"] * np.sin(lG_t) + g_y
    plt.plot([g_x, e_x], [g_y, e_y], color='navy', marker='o')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def spring_moment(td, joint):
    if joint == "j2":
        l23_t = np.radians(td["j2"]["pos"])
        s23_t = np.arctan2(td["j2"]["spg_start_y"] - td["j2"]["spg_end_d"] * np.sin(l23_t), td["j2"]["spg_start_x"] - td["j2"]["spg_end_d"] * np.cos(l23_t))
        f23_t = l23_t - s23_t + np.pi
        rf23 = td["j2"]["spg_end_d"] * np.sin(f23_t)
        s_len = np.linalg.norm([td["j2"]["spg_start_x"] - td["j2"]["spg_end_d"] * np.cos(l23_t),
                            td["j2"]["spg_start_y"] - td["j2"]["spg_end_d"] * np.sin(l23_t)])
        f23 = np.max([s_len - td["j2"]["spg_init_l"], 0]) * td["j2"]["spg_rate"]
        m = rf23 * f23 / 1000 # Convert to Nm
    elif joint == "j3":
        l23_t = np.radians(td["j2"]["pos"])
        l34_t = np.radians(td["j3"]["pos"]) + l23_t
        j3_x = td["l23"]["len"] * np.cos(l23_t)
        j3_y = td["l23"]["len"] * np.sin(l23_t)
        s34_t = np.arctan2(td["j3"]["spg_start_y"] + j3_y - (td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t)),
        td["j3"]["spg_start_x"] + j3_x - (td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)))

        f34_t = l34_t - s34_t + np.pi
        rf34 = (td["j3"]["spg_end_d"] + td["j3"]["spg_end_o"] / np.tan(f34_t)) * np.sin(f34_t)

        s_len = np.linalg.norm([td["j3"]["spg_start_x"] + j3_x - (td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)),
                            td["j3"]["spg_start_y"] + j3_y - (td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t))])

        f34 = np.max([s_len - td["j3"]["spg_init_l"], 0]) * td["j3"]["spg_rate"]
        m = rf34 * f34 / 1000 # Convert to Nm
    return m


def calculate_spring_length(td, joint, angle):
    if joint == "j2":
        l23_t = np.radians(angle)
        s_len = np.linalg.norm([td["j2"]["spg_start_x"] - td["j2"]["spg_end_d"] * np.cos(l23_t),
                                td["j2"]["spg_start_y"] - td["j2"]["spg_end_d"] * np.sin(l23_t)])
    elif joint == "j3":
        l23_t = np.radians(td["j2"]["pos"])
        l34_t = np.radians(angle) + l23_t
        j3_x = td["l23"]["len"] * np.cos(l23_t)
        j3_y = td["l23"]["len"] * np.sin(l23_t)
        s_len = np.linalg.norm([td["j3"]["spg_start_x"] + j3_x - (td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)),
                                td["j3"]["spg_start_y"] + j3_y - (td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t))])
    else:
        raise ValueError(f"Unknown joint: {joint}")
    return s_len



def spring_unfit(td, joint):
    angs = np.linspace(td[joint]["min"], td[joint]["max"], 500)
    lengths = [calculate_spring_length(td, joint, ang) for ang in angs]
    return min(lengths) < td[joint]["spg_init_l"] or max(lengths) > td[joint]["spg_init_l"] * 4 # EXTENSION CONSTANT



def grav_moment(td, joint):
    if joint == "j2":
        l23_t = np.radians(td["j2"]["pos"])
        l34_t = np.radians(td["j3"]["pos"]) + l23_t
        l4G_t = np.radians(td["j5"]["pos"]) + l34_t
        l23_h = td["l23"]["len"] * np.cos(l23_t)
        l34_h = td["l34"]["len"] * np.cos(l34_t)
        l4G_h = td["l4G"]["len"] * np.cos(l4G_t)
        lG_h = td["lG"]["len"] * np.cos(l4G_t)
        m = GRAVITY * (l23_h/2 * td["l23"]["mass"] 
                        + l23_h * td["j3"]["mass"] 
                        + (l23_h + l34_h/2) * td["l34"]["mass"]
                        + (l23_h + l34_h) * td["j5"]["mass"]
                        + (l23_h + l34_h + l4G_h/2) * td["l4G"]["mass"]
                        + (l23_h + l34_h + l4G_h + lG_h/2) * td["lG"]["mass"]) / 1e6 # Convert to Nm
    elif joint == "j3":
        l23_t = np.radians(td["j2"]["pos"])
        l34_t = np.radians(td["j3"]["pos"]) + l23_t
        l4G_t = np.radians(td["j5"]["pos"]) + l34_t
        l34_h = td["l34"]["len"] * np.cos(l34_t)
        l4G_h = td["l4G"]["len"] * np.cos(l4G_t)
        lG_h = td["lG"]["len"] * np.cos(l4G_t)
        m = GRAVITY * ((l34_h/2) * td["l34"]["mass"]
                        + l34_h * td["j5"]["mass"]
                        + (l34_h + l4G_h/2) * td["l4G"]["mass"]
                        + (l34_h + l4G_h + lG_h/2) * td["lG"]["mass"]) / 1e6 # Convert to Nm
    return m


def ik(td, cx, cy, t, links=None):
    l23 = td['l23']['len']
    l34 = td['l34']['len']
    bx = cx - (td['l4G']['len']+td['lG']['len'])*np.cos(t)
    by = cy - (td['l4G']['len']+td['lG']['len'])*np.sin(t)
    l24 = np.sqrt(bx**2+by**2)
    t24 = np.arctan2(by,bx)
    t1 = np.arccos(np.clip((l34**2-l23**2-l24**2)/(-2*l23*l24), -1, 1))+t24
    t2334 = np.arccos(np.clip((l24**2-l34**2-l23**2)/(-2*l34*l23), -1, 1))
    t2 = t1 + t2334 - np.pi
    t3 = 180 - t2334
    ax = bx - l34*np.cos(t2)
    ay = by - l34*np.sin(t2)

    # Arm links
    td['j2']['pos'] = np.degrees(t1)
    td['j3']['pos'] = np.degrees(t2)
    td['j5']['pos'] = np.degrees(t)

    # Spring links
    j3_x = l23*np.cos(t1)
    j3_y = l23*np.sin(t1)
    if links is not None:
        links[0].set_data([0, ax], [0, ay])
        links[1].set_data([ax, bx], [ay, by])
        links[2].set_data([bx, cx], [by, cy])
        links[3].set_data([td["j2"]["spg_start_x"], td["j2"]["spg_end_d"] * np.cos(t1)],
                [td["j2"]["spg_start_y"], td["j2"]["spg_end_d"] * np.sin(t1)])
        links[4].set_data([td["j3"]["spg_start_x"] + j3_x,
                        td["j3"]["spg_end_d"] * np.cos(t2) + j3_x - td["j3"]["spg_end_o"] * np.sin(t2)], 
                        [td["j3"]["spg_start_y"] + j3_y,
                        td["j3"]["spg_end_d"] * np.sin(t2) + j3_y + td["j3"]["spg_end_o"] * np.cos(t2)])



def hotspot_points(cpoint, num_points, size=750, std_dev_p=0.08):
    chosen_points = []
    while len(chosen_points) < num_points:
        batch_size = int((num_points - len(chosen_points)) * 1.5)  # Guessing 1.5 times the required number
        points = np.random.rand(batch_size, 2) * size - size / 2
        distances = np.sqrt(np.sum(points**2, axis=1))
        probabilities = np.exp(-distances**2 / (2 * (size * std_dev_p)**2))
        accepted_points = points[np.random.rand(batch_size) < probabilities]

        chosen_points.extend(accepted_points[:num_points - len(chosen_points)])

    chosen_points = np.array(chosen_points)
    chosen_points[:, 0] += cpoint[0]
    chosen_points[:, 1] += cpoint[1]

    return chosen_points


def update(poses, td, links, moments, ax2, m1, m2, num_poses):
    pt, t = poses

    ik(td, pt[0], pt[1], t, links=links)

    moments.append([spring_moment(td, "j2") - grav_moment(td, "j2"), 
                    - spring_moment(td, "j3") + grav_moment(td, "j3")])

    mh1, mh2 = zip(*moments)
    m1.set_offsets(np.column_stack([range(len(mh1)), mh1]))
    m2.set_offsets(np.column_stack([range(len(mh2)), mh2]))

    y_min = min(min(mh1), min(mh2))
    y_max = max(max(mh1), max(mh2))
    
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim(0, len(mh1))

    if len(moments) == num_poses:
        plt.close()

    return links + [m1, m2]


def simulate(td, point=(300, 150), num_points=50, plot=False):
    moments = [] # j2, j3
    sampled_points = hotspot_points(point, num_points)
    poses = [(point, t) for point in sampled_points for t in np.linspace(np.pi/2, -np.pi/2, 9)]
    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        links = [ax1.plot([], [], color='red', marker='o')[0] for _ in range(5)]
        links[3].set_color('cornflowerblue')
        links[4].set_color('cornflowerblue')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(-150, 500)
        ax1.set_ylim(-100, 400)
        ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], color='blue', s=1)
        
        m1 = ax2.scatter([], [], color='blue', s=1)
        m2 = ax2.scatter([], [], color='red', s=1)
        ax2.legend(['M_J2', 'M_J3'])

        ani = FuncAnimation(fig, update, fargs=(td, links, moments, ax2, m1, m2, len(poses)), frames=poses, init_func=lambda: links + [m1, m2], interval=0.1, repeat=False)
        # ani.save("teacher_sim.gif", writer='pillow', fps=30)
        plt.show()
    else:
        for pose in poses:
            pt, t = pose
            ik(td, pt[0], pt[1], t)
            moments.append([spring_moment(td, "j2") - grav_moment(td, "j2"), 
                    - spring_moment(td, "j3") + grav_moment(td, "j3")])
    return moments



def optimize(td, joint, sr_range, sl_range, sd_range, y_range, x_range=None, steps=5, plot=False):   
        j_sr = td["j3"]["spg_rate"]
        j_sl = td["j3"]["spg_init_l"]
        j_sd = td["j3"]["spg_end_d"]
        j_y = td["j3"]["spg_start_y"]
        j_x = td["j3"]["spg_start_x"]

        sr_values = np.linspace(j_sr - sr_range, j_sr + sr_range, num=steps)
        sl_values = np.linspace(j_sl - sl_range, j_sl + sl_range, num=steps)
        sd_values = np.linspace(j_sd - sd_range, j_sd + sd_range, num=steps)
        y_values = np.linspace(j_y - y_range, j_y + y_range, num=steps)
        print(y_values)
        if x_range is not None:
            x_values = np.linspace(j_x - y_range, j_x + y_range, num=steps)
        else:
            x_values = [0]

        min_mse = float('inf')
        min_sr, min_sl, min_sd, min_y, min_x = None, None, None, None, None

        for sr in tqdm(sr_values):
            for sl in sl_values:
                for sd in sd_values:
                    for y in y_values:
                        for x in x_values:
                            td[joint]["spg_rate"] = sr
                            td[joint]["spg_init_l"] = sl
                            td[joint]["spg_end_d"] = sd
                            td[joint]["spg_start_y"] = y
                            td[joint]["spg_start_x"] = x
                            if spring_unfit(td, joint):
                                tqdm.write("Spring unfit")
                                continue
                            moments = np.array(simulate(td, num_points=500, plot=plot))
                            moment_mse = np.mean(moments**2, axis=0).tolist()
                            tqdm.write(f"SR: {sr}, SL: {sl}, SD: {sd} Y: {y}, X: {x}")
                            tqdm.write(f"J2 MSE: {moment_mse[0]}" if joint == "j2" else f"J3 MSE: {moment_mse[1]}")
                            if moment_mse[1] < min_mse:
                                min_mse = moment_mse[1]
                                min_sr, min_sl, min_sd, min_y, min_x = sr, sl, sd, y, x
        td[joint]["spg_rate"] = min_sr
        td[joint]["spg_init_l"] = min_sl
        td[joint]["spg_end_d"] = min_sd
        td[joint]["spg_start_y"] = min_y
        td[joint]["spg_start_x"] = min_x
        print(f"Range of parameters | SR: {sr_values[0]} - {sr_values[-1]}, SL: {sl_values[0]} - {sl_values[-1]}, SD: {sd_values[0]} - {sd_values[-1]}, Y: {y_values[0]} - {y_values[-1]}, X: {x_values[0]} - {x_values[-1]}")
        print(f"Optimized {joint} parameters | SR: {min_sr}, SL: {min_sl}, SD: {min_sd}, Y: {min_y}, X: {min_x}")
        return td, min_mse


if __name__ == "__main__":
    showplot = True if len(sys.argv) > 2 and sys.argv[2] == "plot" else False
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        teacher_dict = load_json_file(file_name)

        teacher_dict, j3_mse = optimize(teacher_dict, "j3", 0.4, 10, 60, 40, plot=showplot)
        print(f"J3 MSE: {j3_mse}")
        simulate(teacher_dict, num_points=50, plot=True)
        
        teacher_dict, j2_mse = optimize(teacher_dict, "j2", 0.4, 10, 60, 40, 20, plot=showplot)
        print(f"J2 MSE: {j2_mse}")
        simulate(teacher_dict, num_points=50, plot=True)
        
        

    else:
        print("Please provide a file name as a command-line argument.")