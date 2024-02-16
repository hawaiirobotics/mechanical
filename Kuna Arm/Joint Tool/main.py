import json
import sys
<<<<<<< HEAD
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
=======
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import pandas as pd
import signal
from tqdm import tqdm
matplotlib.use('TkAgg')
signal.signal(signal.SIGINT, signal.SIG_DFL)
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461

GRAVITY = 9.81

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def plot(td, pos):
    plt.figure()
    l23_t = np.radians(td["j2"][pos])
    j3_x = td["l23"]["len"] * np.cos(l23_t)
    j3_y = td["l23"]["len"] * np.sin(l23_t)
<<<<<<< HEAD
    plt.plot([td["l23"]["spg_start_x"], td["l23"]["spg_end_d"] * np.cos(l23_t)],
             [td["l23"]["spg_start_y"], td["l23"]["spg_end_d"] * np.sin(l23_t)], color='gray', marker='o')
=======
    plt.plot([td["j2"]["spg_start_x"], td["j2"]["spg_end_d"] * np.cos(l23_t)],
             [td["j2"]["spg_start_y"], td["j2"]["spg_end_d"] * np.sin(l23_t)], color='gray', marker='o')
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
    plt.plot([0, j3_x], [0, j3_y], color='cornflowerblue', marker='o')
    l34_t = np.radians(td["j3"][pos]) + l23_t
    j5_x = td["l34"]["len"] * np.cos(l34_t) + j3_x
    j5_y = td["l34"]["len"] * np.sin(l34_t) + j3_y
<<<<<<< HEAD
    plt.plot([td["l34"]["spg_start_x"] + j3_x, td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)], 
             [td["l34"]["spg_start_y"] + j3_y, td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t)], color='gray', marker='o')
=======
    plt.plot([td["j3"]["spg_start_x"] + j3_x, td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)], 
             [td["j3"]["spg_start_y"] + j3_y, td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t)], color='gray', marker='o')
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
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
<<<<<<< HEAD
        s23_t = np.arctan2(td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t), td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t))
        f23_t = l23_t - s23_t + np.pi
        rf23 = td["l23"]["spg_end_d"] * np.sin(f23_t)
        s_len = np.linalg.norm([td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t),
                            td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t)])
        f23 = np.max([s_len - td["l23"]["spg_init_l"], 0]) * td["l23"]["spg_rate"]
=======
        s23_t = np.arctan2(td["j2"]["spg_start_y"] - td["j2"]["spg_end_d"] * np.sin(l23_t), td["j2"]["spg_start_x"] - td["j2"]["spg_end_d"] * np.cos(l23_t))
        f23_t = l23_t - s23_t + np.pi
        rf23 = td["j2"]["spg_end_d"] * np.sin(f23_t)
        s_len = np.linalg.norm([td["j2"]["spg_start_x"] - td["j2"]["spg_end_d"] * np.cos(l23_t),
                            td["j2"]["spg_start_y"] - td["j2"]["spg_end_d"] * np.sin(l23_t)])
        f23 = np.max([s_len - td["j2"]["spg_init_l"], 0]) * td["j2"]["spg_rate"]
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
        m = rf23 * f23 / 1000 # Convert to Nm
    elif joint == "j3":
        l23_t = np.radians(td["j2"]["pos"])
        l34_t = np.radians(td["j3"]["pos"]) + l23_t
        j3_x = td["l23"]["len"] * np.cos(l23_t)
        j3_y = td["l23"]["len"] * np.sin(l23_t)
<<<<<<< HEAD
        s34_t = np.arctan2(td["l34"]["spg_start_y"] + j3_y - (td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t)),
        td["l34"]["spg_start_x"] + j3_x - (td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)))

        f34_t = l34_t - s34_t + np.pi
        rf34 = (td["l34"]["spg_end_d"] + td["l34"]["spg_end_o"] / np.tan(f34_t)) * np.sin(f34_t)

        s_len = np.linalg.norm([td["l34"]["spg_start_x"] + j3_x - (td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)),
                            td["l34"]["spg_start_y"] + j3_y - (td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t))])

        f34 = np.max([s_len - td["l34"]["spg_init_l"], 0]) * td["l34"]["spg_rate"]
=======
        s34_t = np.arctan2(td["j3"]["spg_start_y"] + j3_y - (td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t)),
        td["j3"]["spg_start_x"] + j3_x - (td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)))

        f34_t = l34_t - s34_t + np.pi
        rf34 = (td["j3"]["spg_end_d"] + td["j3"]["spg_end_o"] / np.tan(f34_t)) * np.sin(f34_t)

        s_len = np.linalg.norm([td["j3"]["spg_start_x"] + j3_x - (td["j3"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["j3"]["spg_end_o"] * np.sin(l34_t)),
                            td["j3"]["spg_start_y"] + j3_y - (td["j3"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["j3"]["spg_end_o"] * np.cos(l34_t))])
        if np.shape(s_len) != ():
            print(np.shape(s_len))
        f34 = np.max([s_len - td["j3"]["spg_init_l"], 0]) * td["j3"]["spg_rate"]
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
        m = rf34 * f34 / 1000 # Convert to Nm
    return m


<<<<<<< HEAD
=======
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
    angs = np.linspace(td[joint]["min"], td[joint]["max"], 50)
    lengths = [calculate_spring_length(td, joint, ang) for ang in angs]
    return min(lengths) < td[joint]["spg_init_l"] or max(lengths) > td[joint]["spg_init_l"] * td[joint]["spg_max_ratio"] # EXTENSION CONSTANT

def spring_min_max(td, joint):
    angs = np.linspace(td[joint]["min"], td[joint]["max"], 50)
    lengths = [calculate_spring_length(td, joint, ang) for ang in angs]
    return min(lengths), max(lengths)


>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
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


<<<<<<< HEAD
def ik(td, cx, cy, t, links):
=======
def ik(td, cx, cy, t, links=None):
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
    l23 = td['l23']['len']
    l34 = td['l34']['len']
    bx = cx - (td['l4G']['len']+td['lG']['len'])*np.cos(t)
    by = cy - (td['l4G']['len']+td['lG']['len'])*np.sin(t)
    l24 = np.sqrt(bx**2+by**2)
    t24 = np.arctan2(by,bx)
<<<<<<< HEAD
    t1 = np.arccos((l34**2-l23**2-l24**2)/(-2*l23*l24))+t24
    t2334 = np.arccos((l24**2-l34**2-l23**2)/(-2*l34*l23))
=======
    t1 = np.arccos(np.clip((l34**2-l23**2-l24**2)/(-2*l23*l24), -1, 1))+t24
    t2334 = np.arccos(np.clip((l24**2-l34**2-l23**2)/(-2*l34*l23), -1, 1))
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
    t2 = t1 + t2334 - np.pi
    t3 = 180 - t2334
    ax = bx - l34*np.cos(t2)
    ay = by - l34*np.sin(t2)

    # Arm links
<<<<<<< HEAD
    links[0].set_data([0, ax], [0, ay])
    links[1].set_data([ax, bx], [ay, by])
    links[2].set_data([bx, cx], [by, cy])
=======
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
    td['j2']['pos'] = np.degrees(t1)
    td['j3']['pos'] = np.degrees(t2)
    td['j5']['pos'] = np.degrees(t)

    # Spring links
    j3_x = l23*np.cos(t1)
    j3_y = l23*np.sin(t1)
<<<<<<< HEAD
    links[3].set_data([td["l23"]["spg_start_x"], td["l23"]["spg_end_d"] * np.cos(t1)],
             [td["l23"]["spg_start_y"], td["l23"]["spg_end_d"] * np.sin(t1)])
    
    links[4].set_data([td["l34"]["spg_start_x"] + j3_x,
                       td["l34"]["spg_end_d"] * np.cos(t2) + j3_x - td["l34"]["spg_end_o"] * np.sin(t2)], 
                       [td["l34"]["spg_start_y"] + j3_y,
                       td["l34"]["spg_end_d"] * np.sin(t2) + j3_y + td["l34"]["spg_end_o"] * np.cos(t2)])


def hotspot_points(cx, cy, num_points, size=750, std_dev_p=0.08):
    chosen_points = np.empty((num_points, 2))

    for i in range(num_points):
        while True:
            point = np.random.rand(2) * size - size/2  # Generate a random point within the grid
            distance = np.sqrt(np.sum(point**2))
            probability = np.exp(-distance**2 / (2 * (size*std_dev_p)**2))

            if np.random.rand() < probability:
                chosen_points[i] = point
                break

    chosen_points[:, 0] += cx
    chosen_points[:, 1] += cy
=======
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
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461

    return chosen_points


<<<<<<< HEAD
def update(frame, td, links, moments, ax2, m1, m2):
    point, t = frame
    ik(td, point[0], point[1], t, links)
=======
def update(poses, td, links, moments, ax2, m1, m2, num_poses):
    pt, t = poses

    ik(td, pt[0], pt[1], t, links=links)
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461

    moments.append([spring_moment(td, "j2") - grav_moment(td, "j2"), 
                    - spring_moment(td, "j3") + grav_moment(td, "j3")])

<<<<<<< HEAD
    print(moments[-1])

=======
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
    mh1, mh2 = zip(*moments)
    m1.set_offsets(np.column_stack([range(len(mh1)), mh1]))
    m2.set_offsets(np.column_stack([range(len(mh2)), mh2]))

    y_min = min(min(mh1), min(mh2))
    y_max = max(max(mh1), max(mh2))
<<<<<<< HEAD
    delta = y_max - y_min
    if y_min < 0:
        y_min = y_min - 1.05 * delta
    else:
        y_min = y_min - 0.95 * delta
    if y_max < 0:
        y_max = y_max + 0.95 * delta
    else:
        y_max = y_max + 1.05 * delta
    
    ax2.set_ylim([y_min, y_max])
=======
    
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim(0, len(mh1))

    if len(moments) == num_poses:
        plt.close()
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461

    return links + [m1, m2]


<<<<<<< HEAD
if __name__ == "__main__":
    if len(sys.argv) > 1:
        moments = [] # j2, j3
        file_name = sys.argv[1]
        teacher_dict = load_json_file(file_name)
        sampled_points = hotspot_points(300, 150, 500)

=======
def simulate(td, point=(300, 150), num_points=50, plot=False):
    moments = [] # j2, j3
    sampled_points = hotspot_points(point, num_points)
    poses = [(point, t) for point in sampled_points for t in np.linspace(np.pi/2, -np.pi/2, 9)]
    if plot:
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        links = [ax1.plot([], [], color='red', marker='o')[0] for _ in range(5)]
        links[3].set_color('cornflowerblue')
        links[4].set_color('cornflowerblue')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(-150, 500)
        ax1.set_ylim(-100, 400)
        ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], color='blue', s=1)
<<<<<<< HEAD
        ax2.set_xlim(0, len(sampled_points) * 8)  # Adjust as needed
        m1 = ax2.scatter([], [], color='blue', s=1)  # For m1
        m2 = ax2.scatter([], [], color='red', s=1) # For m2

        frames = [(point, t) for point in sampled_points for t in np.linspace(np.pi/2, -np.pi/2, 8)]
        ani = FuncAnimation(fig, update, fargs=(teacher_dict, links, moments, ax2, m1, m2), frames=frames, init_func=lambda: links + [m1, m2], blit=True, interval=0.1, repeat=False)

        plt.show()
=======
        
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



def optimize(td, joint, springs, sd_values, y_values, x_values=[0], spg_range=None, plot=False):   
        min_mse = float('inf')
        min_spg, min_sd, min_y, min_x = None, None, None, None

        for index, spg in tqdm(springs.iterrows()):
            for sd in sd_values:
                for y in y_values:
                    for x in x_values:
                        td[joint]["spg_rate"] = spg['Rate N/mm']
                        td[joint]["spg_init_l"] = spg['Lg.']
                        td[joint]["spg_max_ratio"] = spg['Lg. Max']/spg['Lg.']
                        td[joint]["spg_end_d"] = sd
                        td[joint]["spg_start_y"] = y
                        td[joint]["spg_start_x"] = x
                        if spg_range is not None:
                            if spg['Lg.'] < spg_range[joint][sd][y][x][0] or spg['Lg. Max'] > spg_range[joint][sd][y][x][1]:
                                continue
                        else:
                            if spring_unfit(td, joint):
                                continue
                        moments = np.array(simulate(td, num_points=500, plot=plot))
                        moment_mse = np.mean(moments**2, axis=0).tolist()

                        if moment_mse[1] < min_mse:
                            min_mse = moment_mse[1]
                            min_spg, min_sd, min_y, min_x = spg['Part Number'], sd, y, x
        min_spg_data = springs.loc[springs['Part Number'] == min_spg]
        td[joint]["spg"] = min_spg_data["Part Number"].values[0]
        td[joint]["spg_rate"] = min_spg_data["Rate N/mm"].values[0]
        td[joint]["spg_init_l"] = min_spg_data["Lg."].values[0]
        td[joint]["spg_end_d"] = min_sd
        td[joint]["spg_start_y"] = min_y
        td[joint]["spg_start_x"] = min_x
        print(f"Range of parameters SD: {sd_values[0]} - {sd_values[-1]}, Y: {y_values[0]} - {y_values[-1]}, X: {x_values[0]} - {x_values[-1]}")
        print(f"Optimized {joint} parameters | Part Number: {td[joint]['spg']} SR: {td[joint]['spg_rate']}, SL: {td[joint]['spg_init_l']}, SD: {min_sd}, Y: {min_y}, X: {min_x}")
        return td, min_mse


def load_springs(file_path):
    df = pd.read_csv(file_path)

    df['Lg. Max'] = pd.to_numeric(df['Lg. Max'], errors='coerce')
    df['Rate N/mm'] = pd.to_numeric(df['Rate N/mm'], errors='coerce')
    df['Lg.'] = pd.to_numeric(df['Lg.'], errors='coerce')
    df['OD'] = pd.to_numeric(df['OD'], errors='coerce')

    return df


def filter_springs(df, rate_range, length_range, od_range):
    filtered_df = df[
        (df['Rate N/mm'] >= rate_range[0]) & (df['Rate N/mm'] <= rate_range[1]) &
        (df['Lg.'] >= length_range[0]) & (df['Lg.'] <= length_range[1]) &
        (df['OD'] >= od_range[0]) & (df['OD'] <= od_range[1])
    ]
    
    sorted_df = filtered_df.sort_values(by='Rate N/mm')

    return sorted_df

def precompute_ranges(td, j3_sd_values, j3_y_values, j2_sd_values, j2_y_values, j2_x_values):
    t_min_l, t_max_l = float('inf'), 0
    spg_range = {'j3': {}, 'j2': {}}
    for sd in j3_sd_values:
        spg_range['j3'][sd] = {}
        for y in j3_y_values:
            spg_range['j3'][sd][y] = {}  # Corrected from spg_range['j2'][sd][y] to spg_range['j3'][sd][y]
            td['j3']["spg_end_d"] = sd
            td['j3']["spg_start_y"] = y
            min_l, max_l = spring_min_max(td, "j3")
            if min_l < t_min_l:
                t_min_l = min_l
            if max_l > t_max_l:
                t_max_l = max_l
            spg_range['j3'][sd][y][0] = [min_l, max_l]

    for sd in j2_sd_values:
        spg_range['j2'][sd] = {}
        for y in j2_y_values:
            spg_range['j2'][sd][y] = {}
            for x in j2_x_values:
                td['j2']["spg_end_d"] = sd
                td['j2']["spg_start_y"] = y
                td['j2']["spg_start_x"] = x
                min_l, max_l = spring_min_max(td, "j2")
                if min_l < t_min_l:
                    t_min_l = min_l
                if max_l > t_max_l:
                    t_max_l = max_l
                spg_range['j2'][sd][y][x] = [min_l, max_l]
    print(f"Spring length range: {t_min_l} - {t_max_l}")
    return spg_range, t_min_l, t_max_l


def save_json_file(td):
    optimized_file_name = "optimized_teacher.json"
    with open(optimized_file_name, 'w') as outfile:
        json.dump(td, outfile, indent=4)

    print(f"Optimized teacher dictionary saved to {optimized_file_name}")


if __name__ == "__main__":
    showplot = True if len(sys.argv) > 2 and sys.argv[2] == "plot" else False
    PN_search = True if len(sys.argv) > 2 and sys.argv[2] == "search" else False
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        td = load_json_file(file_name)

        steps = 5
        sd_range, y_range, x_range = 60, 30, 40

        springs_df = load_springs('mcmaster_springs.csv')
        print(springs_df)
        if PN_search: # Search for the best spring part number
            j3_sd_values = np.linspace(td["j3"]["spg_end_d"] - sd_range, td["j3"]["spg_end_d"] + sd_range, num=steps)
            j3_y_values = np.linspace(td["j3"]["spg_start_y"] - y_range, td["j3"]["spg_start_y"] + y_range, num=steps)
            j2_sd_values = np.linspace(td["j2"]["spg_end_d"] - sd_range, td["j2"]["spg_end_d"] + sd_range, num=steps)
            j2_y_values = np.linspace(td["j2"]["spg_start_y"] - y_range, td["j2"]["spg_start_y"] + y_range, num=steps)
            j2_x_values = np.linspace(td["j2"]["spg_start_x"] - x_range, td["j2"]["spg_start_x"], num=steps)
            
            spg_range, t_min_l, t_max_l = precompute_ranges(td, j3_sd_values, j3_y_values, j2_sd_values, j2_y_values, j2_x_values)
            
            rate_n_mm_range = (0.1, 0.5)
            length_mm_range = (max(t_min_l, 50), t_max_l)
            od_range = (5, 12)
            filtered_springs = filter_springs(springs_df, rate_n_mm_range, length_mm_range, od_range)

            td, j3_mse = optimize(td, "j3", filtered_springs, j3_sd_values, j3_y_values, spg_range=spg_range, plot=showplot)
            print(f"J3 MSE: {j3_mse}")
            td, j2_mse = optimize(td, "j2", filtered_springs, j2_sd_values, j2_y_values, j2_x_values, spg_range=spg_range, plot=showplot)
            print(f"J2 MSE: {j2_mse}")

        selected_j3_spring = springs_df[springs_df['Part Number'] == td['j3']['spg']]
        selected_j2_spring = springs_df[springs_df['Part Number'] == td['j2']['spg']]
        print(f"Selected J3 Spring: {selected_j3_spring}")
        print(f"Selected J2 Spring: {selected_j2_spring}")

        td = load_json_file(file_name) # Reload the original file
        
>>>>>>> fbc617a917302fa5edc7349d7f8caa84ac05b461

        steps = 20
        j3_sd_values = np.linspace(td["j3"]["spg_end_d"] - sd_range, td["j3"]["spg_end_d"] + sd_range, num=steps)
        j3_y_values = np.linspace(td["j3"]["spg_start_y"] - y_range, td["j3"]["spg_start_y"] + y_range, num=steps)
        j2_sd_values = np.linspace(td["j2"]["spg_end_d"] - sd_range, td["j2"]["spg_end_d"] + sd_range, num=steps)
        j2_y_values = np.linspace(td["j2"]["spg_start_y"] - y_range, td["j2"]["spg_start_y"] + y_range, num=steps)
        j2_x_values = np.linspace(td["j2"]["spg_start_x"] - x_range, td["j2"]["spg_start_x"], num=steps)

        td, j3_mse = optimize(td, "j3", selected_j3_spring, j3_sd_values, j3_y_values, plot=showplot)
        print(f"Reoptimized J3 MSE: {j3_mse}")

        td, j2_mse = optimize(td, "j2", selected_j2_spring, j2_sd_values, j2_y_values, j2_x_values, plot=showplot)
        print(f"Reoptimized J2 MSE: {j2_mse}")

        simulate(td, num_points=50, plot=True)
        print(f"Optimized j3 Mounting: SD: {td['j3']['spg_end_d']}, Y: {td['j3']['spg_start_y']}")
        print(f"Optimized j2 Mounting: SD: {td['j2']['spg_end_d']}, Y: {td['j2']['spg_start_y']} X: {td['j2']['spg_start_x']}")

        save_json_file(td)

    else:
        print("Please provide a file name as a command-line argument.")