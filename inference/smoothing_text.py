from PIL import ImageFont, ImageDraw, Image
import os
import json

dir_name = 'inference_frame_211003'
file_dirs = []
file_names = []
for i in os.listdir("/workspace/jt/places/{}/".format(dir_name)):
    file_dirs.append("/workspace/jt/places/{}/".format(dir_name) + i)
    dir_name = file_names.append(i.split('.')[0])


for i in range(len(file_dirs)):
    VIDEO_NAME = file_names[i]
    print(VIDEO_NAME + " is working")
    base_dir = "/workspace/jt/places/inference_frame_211003/{}/".format(VIDEO_NAME)
    print(base_dir)

    # IMAGE
    img_dir = base_dir + "frame{}/".format(VIDEO_NAME)
    img_list = [_ for _ in os.listdir(img_dir) if _.endswith(r".jpg")]

    for i in range(len(img_list)):

        if i >= len(img_list)-3:
            continue
        image = Image.open(img_dir + img_list[i])

        # JSON
        json_dir = base_dir + "smoothing_{}.json".format(VIDEO_NAME)
        with open(json_dir) as f:
            json_data = json.load(f)


        label_list = []
        top = json_data[i]["description"]
        label_list.append(top)

        label_label = "result : {}".format(label_list[0])

        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype("NanumSquareRoundR.ttf", 30)
        draw.text((10, 10), label_label, fill="red", font=font)

        save_dir = base_dir + "smoothinglabel{}/".format(VIDEO_NAME)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        image.save(save_dir + "{}.jpg".format(i))