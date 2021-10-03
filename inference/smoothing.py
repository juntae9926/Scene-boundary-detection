import json
import os
from collections import deque
import numpy as np

def frameScore(json_dir, label_score, iteration):
    with open(json_dir) as f:
        json_data = json.load(f)

    frame_score = label_score
    for j in range(0, 5):
        top = json_data[iteration]["frame_result"][j]["label"]["description"]
        score = round(json_data[iteration]["frame_result"][j]["label"]["score"], 2)
        frame_score[top] = score
    return frame_score

def labelMap(label_map_path):
    label_map = {}
    with open(label_map_path, 'r', encoding='utf-8') as t:
        for i, line in enumerate(t.readlines()):
            line = line.rstrip('\n')
            label_map[i] = line
    return label_map

def labelScore(label_map_path):
    label_score = {}
    with open(label_map_path, 'r', encoding='utf-8') as t:
        for i, line in enumerate(t.readlines()):
            line = line.rstrip('\n')
            label_score[line] = 0
    return label_score


def main():
    file_dirs = []
    file_names = []
    sevenkeys = []
    label_map = labelMap('/workspace/classes.txt')
    weight = [[0.1], [0.1], [0.15], [0.3], [0.15], [0.1], [0.1]]

    for i in os.listdir("/workspace/jt/places/inference_frame_211003/"):
        file_dirs.append("/workspace/jt/places/inference_frame_211003/" + i)
        dir_name = file_names.append(i.split('.')[0])

    for i in range(len(file_dirs)):
        VIDEO_NAME = file_names[i]
        print(VIDEO_NAME + " is working")
        base_dir = "/workspace/jt/places/inference_frame_211003/{}/".format(VIDEO_NAME)

        # IMAGE
        img_dir = base_dir + "frame{}/".format(VIDEO_NAME)
        img_list = [_ for _ in os.listdir(img_dir) if _.endswith(r".jpg")]
        dq = deque([])

        data = []
        inference_list = [0, 0, 0]

        for i in range(len(img_list)):
            json_dir = base_dir + "{}.json".format(VIDEO_NAME)
            label_score = labelScore('/workspace/classes.txt')

            frame_score = frameScore(json_dir, label_score, i)
            dq.append(frame_score)

            if len(dq) == 7:
                for i in dq:
                    arr1 = list(map(float, (k for k in i.values())))
                    sevenkeys.append(arr1)
                sevenkeys = np.array(sevenkeys)
                sevenkeys.reshape(7, 16)
                mul = np.multiply(sevenkeys, weight)
                mul = np.add.reduce(mul, axis=0)
                max_index = np.argmax(mul)
                inference_list.append(label_map[max_index])
                sevenkeys = []
                dq.popleft()

        for i in range(len(inference_list)):
            result = {"file_number": i, "description": inference_list[i]}
            data.append(result)

        with open(base_dir + '/smoothing_{}.json'.format(VIDEO_NAME), 'w') as outfile:
            json.dump(data, outfile, indent='\t')
        print(data)

if __name__ == '__main__':
    main()