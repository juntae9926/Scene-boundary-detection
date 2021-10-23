import json
import os
from collections import deque
import numpy as np
from math import pi, sqrt, exp

def makeGaussian(n=11, sigma=1):
    r = range(-int(n/2), int(n/2)+1)
    weight = [1/(sigma * np.sqrt(2*pi)) * exp(-float(x)**2/(2*sigma**2)) for x in r]
    weight = np.reshape(weight, (n, 1))
    print("Using weight parameter: ", weight)
    return weight

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


def placeContext(base_dir, VIDEO_NAME):
    with open(base_dir + '/smoothing_{}.json'.format(VIDEO_NAME)) as f:
        json_data = json.load(f)

    data = []
    file_list = []
    result = None

    for i in range(len(json_data)-1):
        if json_data[i]["description"] == json_data[i+1]["description"]:
            file_list.append(i)
            description = json_data[i]["description"]
            result = {"file_context": file_list, "description": description}
            if i == len(json_data)-2:
                data.append(result)

        else:
            file_list.append(i)
            description = json_data[i]["description"]
            result = {"file_context": file_list, "description": description}
            data.append(result)
            file_list = []
            result = {}
    with open(base_dir + '/smoothing_context_{}.json'.format(VIDEO_NAME), 'w') as outfile:
        json.dump(data, outfile, indent='\t')

def main():
    file_dirs = []
    file_names = []
    sevenKeys = []
    label_map = labelMap('/workspace/classes.txt')
    weight = makeGaussian()
    #weight = [[0.1], [0.1], [0.15], [0.3], [0.15], [0.1], [0.1]]
    #weight_2 = [[0.05], [0.075], [0.2], [0.35], [0.2], [0.075], [0.05]]
    #weight_3 = [[0.05], [0.25], [0.4], [0.25], [0.05]]

    for i in os.listdir("/workspace/jt/places/inference_frame_211017/"):
        file_dirs.append("/workspace/jt/places/inference_frame_211017/" + i)
        file_names.append(i.split('/')[-1])
        print(file_names)

    for i in range(len(file_dirs)):
        VIDEO_NAME = file_names[i]
        print(VIDEO_NAME + " is working")
        base_dir = "/workspace/jt/places/inference_frame_211017/{}/".format(VIDEO_NAME)

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

            if len(dq) == len(weight):
                for i in dq:
                    arr1 = list(map(float, (k for k in i.values())))
                    sevenKeys.append(arr1)
                sevenKeys = np.array(sevenKeys)
                sevenKeys.reshape((len(dq), 16))
                mul = np.multiply(sevenKeys, weight)
                mul = np.add.reduce(mul, axis=0)
                max_index = np.argmax(mul)
                inference_list.append(label_map[max_index])
                sevenKeys = []
                dq.popleft()

        for i in range(len(inference_list)):
            result = {"file_number": i, "description": inference_list[i]}
            data.append(result)

        with open(base_dir + '/smoothing_{}.json'.format(VIDEO_NAME), 'w') as outfile:
            json.dump(data, outfile, indent='\t')
        placeContext(base_dir, VIDEO_NAME)
        print(data)

if __name__ == '__main__':
    main()