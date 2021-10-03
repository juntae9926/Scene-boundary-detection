import numpy as np
from PIL import ImageFont, ImageDraw, Image
import os
import json
import cv2

dir_name = 'demo_frame_210904'
file_dirs = []
file_names = []
for i in os.listdir("/workspace/jt/places/{}/".format(dir_name)):
    file_dirs.append("/workspace/jt/places/{}/".format(dir_name) + i)
    dir_name = file_names.append(i.split('.')[0])


for i in range(len(file_dirs)):
    VIDEO_NAME = file_names[i]
    print(VIDEO_NAME + " is working")
    base_dir = "/workspace/jt/places/demo_frame_210904/{}/".format(VIDEO_NAME)
    print(base_dir)

    # IMAGE
    img_dir = base_dir + "frame{}/".format(VIDEO_NAME)
    img_list = [_ for _ in os.listdir(img_dir) if _.endswith(r".jpg")]

    for i in range(len(img_list)):
        image = Image.open(img_dir + img_list[i])

        # JSON
        json_dir = base_dir + "{}.json".format(VIDEO_NAME)
        with open(json_dir) as f:
            json_data = json.load(f)


        label_list = []
        score_list = []
        for j in range(0, 5):
            top = json_data[i]["frame_result"][j]["label"]["description"]
            score = round(json_data[i]["frame_result"][j]["label"]["score"], 2)
            label_list.append(top)
            score_list.append(score)

        label_label = " top1: {}\n top2: {}\n top3: {}\n top4: {}\n top5: {}"\
            .format(label_list[0], label_list[1], label_list[2],
                    label_list[3], label_list[4])

        label_score = " score: {}\n score: {}\n score: {}\n score: {}\n score: {}"\
            .format(score_list[0], score_list[1], score_list[2],
                    score_list[3], score_list[4])

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("NanumSquareRoundR.ttf", 20)
        draw.text((10, 10), label_label, fill="yellow", font=font)

        draw_score = ImageDraw.Draw(image)
        font = ImageFont.truetype("NanumSquareRoundR.ttf", 20)
        draw_score.text((200, 10), label_score, fill="yellow", font=font)

        save_dir = base_dir + "label{}/".format(VIDEO_NAME)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image.save(save_dir + "{}.png".format(i))





