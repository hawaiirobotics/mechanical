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

def spring_moment_j2(td):
    l23_t = np.radians(td["j2"]["def"])
    s23_t = np.arctan2(td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t), td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t))
    f23_t = l23_t - s23_t + np.pi
    rf23 = td["l23"]["spg_end_d"] * np.sin(f23_t)
    s_len = np.linalg.norm([td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t),
                           td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t)])
    f23 = np.max([s_len - td["l23"]["spg_init_l"], 0]) * td["l23"]["spg_rate"]
    m2 = rf23 * f23 / 1000 # Convert to Nm
    return m2

def grav_moment_j2(td):
    l23_t = np.radians(td["j2"]["def"])
    l34_t = np.radians(td["j3"]["def"]) + l23_t
    l4G_t = np.radians(td["j5"]["def"]) + l34_t
    l23_h = td["l23"]["len"] * np.cos(l23_t)
    l34_h = td["l34"]["len"] * np.cos(l34_t)
    l4G_h = td["l4G"]["len"] * np.cos(l4G_t)
    lG_h = td["lG"]["len"] * np.cos(l4G_t)
    m2 = GRAVITY * (l23_h/2 * td["l23"]["mass"] 
                    + l23_h * td["j3"]["mass"] 
                    + (l23_h + l34_h/2) * td["l34"]["mass"]
                    + (l23_h + l34_h) * td["j5"]["mass"]
                    + (l23_h + l34_h + l4G_h/2) * td["l4G"]["mass"]
                    + (l23_h + l34_h + l4G_h + lG_h/2) * td["lG"]["mass"]) / 1e6 # Convert to Nm
    return m2

def spring_moment_j3(td):
    l23_t = np.radians(td["j2"]["def"])
    l34_t = np.radians(td["j3"]["def"]) + l23_t
    j3_x = td["l23"]["len"] * np.cos(l23_t)
    j3_y = td["l23"]["len"] * np.sin(l23_t)
    s34_t = np.arctan2(td["l34"]["spg_start_y"] + j3_y - (td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t)),
    td["l34"]["spg_start_x"] + j3_x - (td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)))

    f34_t = l34_t - s34_t + np.pi
    rf34 = (td["l34"]["spg_end_d"] + td["l34"]["spg_end_o"] / np.tan(f34_t)) * np.sin(f34_t)

    s_len = np.linalg.norm([td["l34"]["spg_start_x"] + j3_x - (td["l34"]["spg_end_d"] * np.cos(l34_t) + j3_x - td["l34"]["spg_end_o"] * np.sin(l34_t)),
                           td["l34"]["spg_start_y"] + j3_y - (td["l34"]["spg_end_d"] * np.sin(l34_t) + j3_y + td["l34"]["spg_end_o"] * np.cos(l34_t))])

    f34 = np.max([s_len - td["l34"]["spg_init_l"], 0]) * td["l34"]["spg_rate"]
    m3 = rf34 * f34 / 1000 # Convert to Nm
    return m3

def grav_moment_j3(td):
    l23_t = np.radians(td["j2"]["def"])
    l34_t = np.radians(td["j3"]["def"]) + l23_t
    l4G_t = np.radians(td["j5"]["def"]) + l34_t
    l34_h = td["l34"]["len"] * np.cos(l34_t)
    l4G_h = td["l4G"]["len"] * np.cos(l4G_t)
    lG_h = td["lG"]["len"] * np.cos(l4G_t)
    m3 = GRAVITY * ((l34_h/2) * td["l34"]["mass"]
                    + l34_h * td["j5"]["mass"]
                    + (l34_h + l4G_h/2) * td["l4G"]["mass"]
                    + (l34_h + l4G_h + lG_h/2) * td["lG"]["mass"]) / 1e6 # Convert to Nm
    return m3

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

    links[0].set_data([0, ax], [0, ay])
    links[1].set_data([ax, bx], [ay, by])
    links[2].set_data([bx, cx], [by, cy])

    # Spring links
    j3_x = l23*np.cos(t1)
    j3_y = l23*np.sin(t1)
    links[3].set_data([td["l23"]["spg_start_x"], td["l23"]["spg_end_d"] * np.cos(t1)],
             [td["l23"]["spg_start_y"], td["l23"]["spg_end_d"] * np.sin(t1)])
    
    links[4].set_data([td["l34"]["spg_start_x"] + j3_x,
                       td["l34"]["spg_end_d"] * np.cos(t2) + j3_x - td["l34"]["spg_end_o"] * np.sin(t2)], 
                       [td["l34"]["spg_start_y"] + j3_y,
                       td["l34"]["spg_end_d"] * np.sin(t2) + j3_y + td["l34"]["spg_end_o"] * np.cos(t2)])




def update(frame, links):
    point, t = frame
    ik(teacher_dict, point[0], point[1], t, links)
    return links


def hotspot_sim(cx, cy, num_points, size=750, std_dev_p=0.08):
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

if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        teacher_dict = load_json_file(file_name)
        # ik(teacher_dict,500,150,0.7)
        sampled_points = hotspot_sim(300, 150, 500)

        fig, ax = plt.subplots()
        links = [ax.plot([], [], color='red', marker='o')[0] for _ in range(5)]
        links[3].set_color('cornflowerblue')
        links[4].set_color('cornflowerblue')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(-150, 500)
        ax.set_ylim(-100, 400)
        ax.scatter(sampled_points[:, 0], sampled_points[:, 1], color='blue', s=1)
        frames = [(point, t) for point in sampled_points for t in np.linspace(np.pi/2, -np.pi/2, 8)]

        # Start the animation
        ani = FuncAnimation(fig, update, fargs=(links,), frames=frames, init_func=lambda: links, blit=True, interval=20, repeat=False)

        plt.show()
        '''    
        j2_def = teacher_dict["j2"]["def"]
        j2_min = teacher_dict["j2"]["min"]
        j3_def = teacher_dict["j3"]["def"]
        j3_max = teacher_dict["j3"]["max"]

        j2_spring_moments = []
        j2_gravity_moments = []
        j3_spring_moments = []
        j3_gravity_moments = []

        j2_traj = list(range(int(j2_def), int(j2_min)+1, -1))
        j3_traj = list(range(int(j3_def), int(j3_max)-1, 2))
        print(j2_traj)
        print(j3_traj)
        for i in range(0,len(j2_traj)):
            teacher_dict["j2"]["def"] = j2_traj[i]
            teacher_dict["j3"]["def"] = j3_traj[i]
            print("J2: ",teacher_dict["j2"]["def"])
            print("J3: ",teacher_dict["j3"]["def"])
            if teacher_dict["j3"]["def"] > j3_max or teacher_dict["j2"]["def"] < j2_min:
                break
            j2_spring_moments.append(spring_moment_j2(teacher_dict))
            j2_gravity_moments.append(grav_moment_j2(teacher_dict))
            j3_spring_moments.append(spring_moment_j3(teacher_dict))
            j3_gravity_moments.append(grav_moment_j3(teacher_dict))

            # plot(teacher_dict, "def")

        plt.figure()
        plt.plot(j2_traj, j2_spring_moments, color='blue', label='J2 Spring Moments')
        plt.plot(j2_traj, j2_gravity_moments, color='lightblue', label='J2 Gravity Moments')
        plt.plot(j2_traj, j3_spring_moments, color='red', label='J3 Spring Moments')
        plt.plot(j2_traj, j3_gravity_moments, color='salmon', label='J3 Gravity Moments')
        plt.title('J2 and J3 Moments')
        plt.xlabel('Arm Position')
        plt.ylabel('Moment (Nm)')
        plt.legend()
        plt.show()
        '''
    else:
        print("Please provide a file name as a command-line argument.")