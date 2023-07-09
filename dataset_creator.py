import os
import cv2
import json

result_path = "custom_dataset.json"
videos_root = "./videos/"

class_data = [0]*16 + [1]*14 + [2]*15 + [3]*23
subset_data = ["train"]*68

subset_data[6] = "test"
subset_data[9] = "test"
subset_data[11] = "test"

subset_data[22] = "test"
subset_data[25] = "test"
subset_data[29] = "test"

subset_data[36] = "test"
subset_data[39] = "test"
subset_data[42] = "test"

subset_data[50] = "test"
subset_data[57] = "test"
subset_data[64] = "test"



data_dict = {}

index = 0
for name in os.listdir(videos_root):
    print(name)
    video = cv2.VideoCapture(videos_root + name)
    internal_dict = {}
    length = 0
    while video.isOpened():
        ret, frame = video.read()
        length += 1
        if not ret:
            break
    internal_dict["subset"] = subset_data[index]
    internal_dict["action"] = [class_data[index], 1, length]
    data_dict[name[:-4]] = internal_dict
    index += 1

res = json.dumps(data_dict)

result_file = open(result_path, "w")
result_file.write(res)
result_file.close()