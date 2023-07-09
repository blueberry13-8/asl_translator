from tqdm import tqdm
import cv2
import os
import torch
import json
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

videos_root = "./custom_dataset/videos"
result_root = "./custom_dataset/hands_point_arrays"
hands_detector = HandDetector()
pose_detector = PoseDetector()

IGNORE_NO_HANDS = False # an important thing!


videos_js = json.load(open("./custom_dataset/custom_dataset.json"))

video_names = [name for name in os.listdir(videos_root)]

bar = tqdm(video_names)
file_number = len(videos_js.keys())
index = 0

for name in bar:
    full_path = videos_root + "/" + name
    video = cv2.VideoCapture(full_path)
    frames_points = []
    frame_cnt = 0
    r_name = name[:-4]
    if videos_js.get(r_name) is None:
        continue

    start_frame, end_frame = videos_js[r_name]["action"][1], videos_js[r_name]["action"][2]
    while video.isOpened():
        ret, frame = video.read()
        frame_cnt += 1
        # If frame inside action frames then preprocess them
        if ret and start_frame <= frame_cnt <= end_frame:
            # Collect all points. 21 points for each hand, 33 points on pose
            points = [0] * (21 * 3 * 2 + 33 * 3)
            # Recognize hands and collect them into list of all points
            hands, img1 = hands_detector.findHands(frame)
            for i in range(len(hands)):
                ind_shift = 0
                if hands[i].get('type') == 'Left':
                    ind_shift = 21 * 3
                hand_points = hands[i].get('lmList')
                for j in range(len(hand_points)):
                    for k in range(3):
                        points[ind_shift + j * 3 + k] = hand_points[j][k]
            
            # Check if no hands detected
            if IGNORE_NO_HANDS == True:
                all_zeros = True
                for i in range(0, 21*3*2):
                    if points[i] != 0:
                        all_zeros = False
                        break
                if all_zeros == True:
                    continue
            # Recognize the pose and collect points
            img2 = pose_detector.findPose(frame)
            lmList, bboxInfo = pose_detector.findPosition(frame, bboxWithHands=False)
            for i in range(len(lmList)):
                for j in range(1, 4):
                    points[21 * 3 * 2 + i * 3 + j - 1] = lmList[i][j]
            frames_points.append(points)
        elif not ret:
            break

    # Release the video object
    video.release()

    # Convert the list of frames to a PyTorch tensor
    tensor = torch.tensor(frames_points)
    torch.save(tensor, result_root + "/" + r_name + ".pt")
    index += 1
    bar.set_postfix(({"Files loaded": f"{index} of {file_number}"}))
