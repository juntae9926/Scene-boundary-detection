import torch
import torchvision.transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import json
import os

dir_name = 'inference_frame_211003'
model_name = '20211003-15.pth.tar'

# GPU device setting
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# Initial arguments
topk = 5
SKIP_FRAME = 30
file_name = None
model_path = '/workspace/jt/model/{}'.format(model_name)
label_map_path = '/workspace/jt/places/places_16_210904/classes.txt'.format(dir_name)

# MatricTracker편에서 만들어 두었던 label_map 메서드를 활용.
from metrictracker import label_map
label_map = label_map(label_map_path)
state = torch.load(model_path)

file_dirs = []
file_names = []
for i in os.listdir("/workspace/jt/places/{}/".format(dir_name)):
    file_dirs.append("/workspace/jt/places/{}/".format(dir_name) + i)
    file_name = file_names.append(i.split('.')[0])
print("file dirs: ", file_dirs)
print("file names: ", file_names)

##### DATA TRANSFORM #####
mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
transforms = torchvision.transforms.Compose([
    torchvision.transforms.RandomResizedCrop((224, 224)),
    torchvision.transforms.ToTensor(),  # Tensor 형태로 변경
    # transforms.Normalize(mean, std)
])  # 0~255의 픽셀 값을 0~1로 변경

# Inference Start #
if __name__ == '__main__':
    for i in range(len(file_dirs)):
        root_dir = file_dirs[i]
        file_name = file_names[i]
        print(root_dir)

        datasets = ImageFolder(os.path.join(root_dir), transform=transforms, target_transform=None)
        valloader = DataLoader(datasets, batch_size=1, shuffle=False, num_workers=1)

        import models
        # wideresnet의 model 가져오기
        model = models.resnet50()
        model.load_state_dict(state['state_dict'], strict=False)
        model = model.to(device)

        data = []

        allFiles, _ = map(list, zip(*valloader.dataset.samples))

        for i, (input, label) in enumerate(valloader):
            model.eval()
            #print("Image index is {}".format(i + 1))

            input, label = input.to(device), label.to(device)
            #print("What is input:", input)

            output = model(input)
            #print("out", output)
            loss = F.softmax(output, dim=1).data.squeeze()
            probs, idx = loss.sort(0, True)
            probs = probs.cpu().numpy()
            idx = idx.cpu().numpy()

            result = {"file_number": i, "frame_number": i * SKIP_FRAME, "frame_result": []}
            # frame_url, timestamp 필요 시 추가
            # result["frame_url"] = allFiles[i]
            # result["timestamp"] = "{}-s".format(i)

            for j in range(0, topk):
                label = {'label': {
                    'description': label_map[idx[j]],
                    'score': float(probs[j]) * 100,
                }}
                result['frame_result'].append(label)
            data.append(result)

            with open(root_dir+'/{}.json'.format(file_name), 'w') as outfile:
                json.dump(data, outfile, indent='\t')
                #print(json.dumps(data, indent='\t'))

            ### 원래 정답과 비교하여 True/False 출력 ###
            #print("Where is this place? ",
            #      label_map[idx[0]])  # ImageFolder로 디렉토리 idx를 읽어 classex.txt 딕셔너리로부터 ground truth 확보 (for validation)
            #label_idx = label.cpu().numpy()[0]
            #label_name = label_map[label_idx]
            #print("Where is ground truth place? ", label_name)
            #if label_name == label_map[idx[0]]:  # directory name과 inference class가 같음?
            #    print("True!!!")
            #else:
            #    print("False!!!")

            #print('Prob of {}: {:.2f}%'.format(label_map[idx[0]], 100 * probs[0]))  # confidence 확률이 나옴
            #print("-" * 10)

# json파일 자동생성 및 자동 저장하게끔 디렉토리 이름에서 따오기
# resnet-101로 실험 ㄱㄱ
