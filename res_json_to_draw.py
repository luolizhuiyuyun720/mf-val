import pickle
import json
import numpy as np

def load_json_file(json_file_path):
    with open(json_file_path, 'r') as fin:
        o = json.load(fin)
    return o

def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period

def ConvertJson(input_json, output_json):
    frame_count = len(input_json)
    if frame_count == 0:
        return

    new_json = input_json

    for k in range(frame_count):
        if new_json[k] is None:
            continue

        obs = len(new_json[k]['dimensions'])
        for n in range(obs):
            w, l, h = new_json[k]['dimensions'][n][0], new_json[k]['dimensions'][n][1], new_json[k]['dimensions'][n][2]
            x, y, z = new_json[k]['location'][n][0], new_json[k]['location'][n][1], new_json[k]['location'][n][2]
            vx, vy = new_json[k]['velocity'][n][0], new_json[k]['velocity'][n][1]

            new_json[k]['dimensions'][n][0] = h
            new_json[k]['dimensions'][n][1] = w
            new_json[k]['dimensions'][n][2] = l

            new_json[k]['location'][n][1] = -y

            r_y = new_json[k]['rotation_y'][n] + np.pi/2
            new_json[k]['rotation_y'][n] = limit_period(r_y)

            # new_json[k]['velocity'][n][1] = -vy

    json.dump(new_json, output_json, indent=2)

if __name__ == '__main__':
    input_json_file = '/home/lynn/Work/apollo-dev/dataset/rt-res.json'
    output_json_file = '/media/lynn/B2EE9C02EE9BBD53/mf/WANGSHUO/August1/cpp_dt2.json'

    input = load_json_file(input_json_file)
    output = open(output_json_file, 'w')

    ConvertJson(input, output)


