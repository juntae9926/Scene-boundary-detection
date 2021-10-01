import json
import os
from PIL import ImageFont, ImageDraw, Image

file_dirs = []
file_names = []
for i in os.listdir("/workspace/jt/places/demo_frame_korean/"):
    file_dirs.append("/workspace/jt/places/demo_frame_korean/" + i)
    file_name = file_names.append(i.split('.')[0])

effective_files = []
effective_labels = []
top1_label_list = []

for i in range(len(file_dirs)):
    VIDEO_NAME = file_names[i]
    print(VIDEO_NAME + " is working")
    base_dir = "/workspace/jt/places/demo_frame_korean/{}/".format(VIDEO_NAME)

    # IMAGE
    img_dir = base_dir + "frame{}/".format(VIDEO_NAME)
    img_list = [_ for _ in os.listdir(img_dir) if _.endswith(r".jpg")]

    for i in range(len(img_list)):
        image = Image.open(img_dir + img_list[i])

        # JSON
        json_dir = base_dir + "{}.json".format(VIDEO_NAME)
        with open(json_dir) as f:
            json_data = json.load(f)

        file_number = json_data[i]["file_number"]

        label_list = []
        score_list = []
        for j in range(0, 5):
            top = json_data[i]["frame_result"][j]["label"]["description"]
            score = round(json_data[i]["frame_result"][j]["label"]["score"], 2)
            label_list.append(top)
            score_list.append(score)

        top1_label_list.append(label_list[0])

        if score_list[0] > 60:
            effective_files.append(file_number)
            effective_labels.append(label_list[0])

        save_dir = base_dir + "label{}/".format(VIDEO_NAME)

count = {}
for i in effective_labels:
    try: count[i] += 1
    except: count[i] = 1

count_all = {}
for j in top1_label_list:
    try: count_all[j] += 1
    except: count_all[j] = 1

print(count)
print(count_all)
print(effective_files)
print(effective_labels)

