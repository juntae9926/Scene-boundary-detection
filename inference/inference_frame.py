import torch
import torch.nn.functional as F
import cv2
from metrictracker import label_map
import json

def classification_img(img_path, model):

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    img = cv2.imread(img_path)
    cv2.imshow('image', img)
    print(img)
    img = cv2.resize(img, dsize=(224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.tensor(img).float()
    img = img.permute(2, 0, 1).unsqueeze(0) # (H, W, C) -> (C, H, W) -> (1, C, H, W)
    print(img)


    model.eval()
    img = img.to(device)

    out = model(img)
    print("out:", out)
    loss = F.softmax(out, dim=1).data.squeeze()
    print("Softmax:", loss)
    probs, idx = loss.sort(0, True)
    probs = probs.cpu().numpy()
    idx = idx.cpu().numpy()
    print(probs, idx)
    return probs, idx

def makeJson(dir, idx, probs, topk=5):
    data=[]
    result = {"frame_name": dir.split('/')[-1], "frame_result": []}
    for j in range(0, topk):
        label = {'label': {
            'description': label_map[idx[j]],
            'score': float(probs[j]) * 100,
        }}
        result['frame_result'].append(label)
    data.append(result)

    return data


if __name__ == '__main__':
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/workspace/jt/places/test_video/001/frame001/', help='Dir Inference_Img')
    parser.add_argument('--model_path', default='/workspace/jt/model/train20210719-17.pth.tar', help='Path checkpoint_model')
    parser.add_argument('--label_map_path', default='/workspace/jt/places/places_17_210713/classes.txt', help='Path label_map')

    args = parser.parse_args()

    infer_img_dir = args.input_dir
    model_path = args.model_path
    label_map_path = args.label_map_path

    # MatricTracker편에서 만들어 두었던 label_map 메서드를 활용.
    label_map = label_map(label_map_path)
    state = torch.load(model_path)

    # wideresnet의 model 가져오기
    import wideresnet
    model = wideresnet.resnet101()
    model.load_state_dict(state['state_dict'], strict=False)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    file_dirs = []
    for i in os.listdir(infer_img_dir):
        file_dirs.append(infer_img_dir + i)

    for dir in file_dirs:
        probs, idx = classification_img(dir, model)
        data = makeJson(dir, idx=idx, probs=probs)
        print('Prob of {}: {:.2f}%'.format(label_map[idx[0]], 100 * probs[0]))  # confidence 확률이 나옴
        print("-" * 10)

        file_name = dir.split('/')[-1]
        json_name = file_name.split('.')[-2]
        print("file save dir :", json_name)
        with open(infer_img_dir + '/{}.json'.format(json_name), 'w') as outfile:
            json.dump(data, outfile, indent='\t')
            print(json.dumps(data, indent='\t'))
