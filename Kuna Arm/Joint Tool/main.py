import json
import sys
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np

GRAVITY = 9.81

def load_json_file(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)


def plot(td, pos):
    plt.figure()
    l23_t = np.radians(td["j2"][pos])
    j3_x = td["l23"]["len"] * np.cos(l23_t)
    j3_y = td["l23"]["len"] * np.sin(l23_t)
    plt.plot([td["l23"]["spg_start_x"], td["l23"]["spg_end_d"] * np.cos(l23_t)],
             [td["l23"]["spg_start_y"], td["l23"]["spg_end_d"] * np.sin(l23_t)], color='gray', marker='o')
    plt.plot([0, j3_x], [0, j3_y], color='cornflowerblue', marker='o')
    l34_t = np.radians(td["j3"][pos]) + l23_t
    j5_x = td["l34"]["len"] * np.cos(l34_t) + j3_x
    j5_y = td["l34"]["len"] * np.sin(l34_t) + j3_y
    plt.plot([td["l34"]["spg_start_x"] + j3_x, td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)], 
             [td["l34"]["spg_start_y"] + j3_y, td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t)], color='gray', marker='o')
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
        s23_t = np.arctan2(td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t), td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t))
        f23_t = l23_t - s23_t + np.pi
        rf23 = td["l23"]["spg_end_d"] * np.sin(f23_t)
        s_len = np.linalg.norm([td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t),
                            td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t)])
        f23 = np.max([s_len - td["l23"]["spg_init_l"], 0]) * td["l23"]["spg_rate"]
        m = rf23 * f23 / 1000 # Convert to Nm
    elif joint == "j3":
        l23_t = np.radians(td["j2"]["pos"])
        l34_t = np.radians(td["j3"]["pos"]) + l23_t
        j3_x = td["l23"]["len"] * np.cos(l23_t)
        j3_y = td["l23"]["len"] * np.sin(l23_t)
        s34_t = np.arctan2(td["l34"]["spg_start_y"] + j3_y - (td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t)),
        td["l34"]["spg_start_x"] + j3_x - (td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)))

        f34_t = l34_t - s34_t + np.pi
        rf34 = (td["l34"]["spg_end_d"] + td["l34"]["spg_end_o"] / np.tan(f34_t)) * np.sin(f34_t)

        s_len = np.linalg.norm([td["l34"]["spg_start_x"] + j3_x - (td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)),
                            td["l34"]["spg_start_y"] + j3_y - (td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t))])

        f34 = np.max([s_len - td["l34"]["spg_init_l"], 0]) * td["l34"]["spg_rate"]
        m = rf34 * f34 / 1000 # Convert to Nm
    return m


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


def ik(td, cx, cy, t, links):
    l23 = td['l23']['len']
    l34 = td['l34']['len']
    bx = cx - (td['l4G']['len']+td['lG']['len'])*np.cos(t)
    by = cy - (td['l4G']['len']+td['lG']['len'])*np.sin(t)
    l24 = np.sqrt(bx**2+by**2)
    t24 = np.arctan2(by,bx)
    t1 = np.arccos((l34**2-l23**2-l24**2)/(-2*l23*l24))+t24
    t2334 = np.arccos((l24**2-l34**2-l23**2)/(-2*l34*l23))
    t2 = t1 + t2334 - np.pi
    t3 = 180 - t2334
    ax = bx - l34*np.cos(t2)
    ay = by - l34*np.sin(t2)

    # Arm links
    links[0].set_data([0, ax], [0, ay])
    links[1].set_data([ax, bx], [ay, by])
    links[2].set_data([bx, cx], [by, cy])
    td['j2']['pos'] = np.degrees(t1)
    td['j3']['pos'] = np.degrees(t2)
    td['j5']['pos'] = np.degrees(t)

    # Spring links
    j3_x = l23*np.cos(t1)
    j3_y = l23*np.sin(t1)
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

    return chosen_points


def update(frame, td, links, moments, ax2, m1, m2):
    point, t = frame
    ik(td, point[0], point[1], t, links)

    moments.append([spring_moment(td, "j2") - grav_moment(td, "j2"), 
                    - spring_moment(td, "j3") + grav_moment(td, "j3")])

    print(moments[-1])

    mh1, mh2 = zip(*moments)
    m1.set_offsets(np.column_stack([range(len(mh1)), mh1]))
    m2.set_offsets(np.column_stack([range(len(mh2)), mh2]))

    y_min = min(min(mh1), min(mh2))
    y_max = max(max(mh1), max(mh2))
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

    return links + [m1, m2]


if __name__ == "__main__":
    if len(sys.argv) > 1:
        moments = [] # j2, j3
        file_name = sys.argv[1]
        teacher_dict = load_json_file(file_name)
        sampled_points = hotspot_points(300, 150, 500)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        links = [ax1.plot([], [], color='red', marker='o')[0] for _ in range(5)]
        links[3].set_color('cornflowerblue')
        links[4].set_color('cornflowerblue')
        ax1.set_aspect('equal', adjustable='box')
        ax1.set_xlim(-150, 500)
        ax1.set_ylim(-100, 400)
        ax1.scatter(sampled_points[:, 0], sampled_points[:, 1], color='blue', s=1)
        ax2.set_xlim(0, len(sampled_points) * 8)  # Adjust as needed
        m1 = ax2.scatter([], [], color='blue', s=1)  # For m1
        m2 = ax2.scatter([], [], color='red', s=1) # For m2

        frames = [(point, t) for point in sampled_points for t in np.linspace(np.pi/2, -np.pi/2, 8)]
        ani = FuncAnimation(fig, update, fargs=(teacher_dict, links, moments, ax2, m1, m2), frames=frames, init_func=lambda: links + [m1, m2], blit=True, interval=0.1, repeat=False)

        plt.show()

    else:
        print("Please provide a file name as a command-line argument.")