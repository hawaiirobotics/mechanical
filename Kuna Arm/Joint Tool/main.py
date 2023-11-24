import json
import sys
import matplotlib.pyplot as plt
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

def forces(td, pos):
    l23_t = np.radians(td["j2"][pos])
    s23_t = np.arctan2(td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t), td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t))
    f23_t = l23_t - s23_t
    rf23 = td["l23"]["spg_end_d"] * np.sin(f23_t)
    f23 = (np.linalg.norm([td["l23"]["spg_start_x"] - td["l23"]["spg_end_d"] * np.cos(l23_t), td["l23"]["spg_start_y"] - td["l23"]["spg_end_d"] * np.sin(l23_t)]) - td["l23"]["spg_init_l"]) * td["l23"]["spg_rate"]
    m23 = rf23 * f23
    print(m23)
    



if __name__ == "__main__":
    if len(sys.argv) > 1:
        file_name = sys.argv[1]
        teacher_dict = load_json_file(file_name)
        forces(teacher_dict,"def")
        plot(teacher_dict,"def")
    else:
        print("Please provide a file name as a command-line argument.")