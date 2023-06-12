import json
import os

import torch
from torch.utils.data import Dataset
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector
import cv2


class ASLDataset(Dataset):
    """
    A custom dataset class for loading American Sign Language (ASL) videos and their corresponding labels.

    Args:
        video_folder (str): Path to the folder containing the ASL videos.
        name_with_label (dict): A dictionary mapping video names to their labels.
        classes (list): A list of class names.
        transform (callable, optional): A function/transform to apply to the frames of the videos.

    Attributes:
        video_folder (str): Path to the folder containing the ASL videos.
        name_with_label (dict): A dictionary mapping video names to their labels.
        videos_names (list): A list of video names.
        classes (list): A list of class names.
        transform (callable, optional): A function/transform to apply to the frames of the videos.
        hands_detector (HandDetector): An instance of the HandDetector class for hand detection.
        pose_detector (PoseDetector): An instance of the PoseDetector class for pose detection.
    """

    def __init__(self, video_folder, name_with_label, classes, transform=None):
        self.video_folder = video_folder
        self.name_with_label = name_with_label
        self.videos_names = list(self.name_with_label.keys())
        self.classes = classes
        self.transform = transform
        self.hands_detector = HandDetector()
        self.pose_detector = PoseDetector()

    def __len__(self):
        """
        Returns the number of videos in the dataset.

        Returns:
            int: The number of videos in the dataset.
        """
        return len(self.videos_names)

    def __getitem__(self, index):
        """
        Retrieves a video and its corresponding label from the dataset.

        Args:
            index (int): The index of the video to retrieve.

        Returns:
            tuple: A tuple containing the video frames as a PyTorch tensor and the label.
        """
        video_path = self.video_folder + '/' + self.videos_names[index] + '.mp4'
        # Open the video file using OpenCV
        video = cv2.VideoCapture(video_path)
        frames_points = []
        frame_cnt = 0
        start_frame, end_frame = self.name_with_label[self.videos_names[index]][1], self.name_with_label[self.videos_names[index]][2]
        while video.isOpened():
            ret, frame = video.read()
            frame_cnt += 1
            # If frame inside action frames then preprocess them
            if ret and start_frame <= frame_cnt <= end_frame:
                # Perform any necessary preprocessing on the frame
                if self.transform is not None:
                    frame = self.transform(frame)
                # Collect all points. 21 points for each hand, 33 points on pose
                points = [0] * (21 * 3 * 2 + 33 * 3)

                # Recognize hands and collect them into list of all points
                hands, img1 = self.hands_detector.findHands(frame)
                for i in range(len(hands)):
                    ind_shift = 0
                    if hands[i].get('type') == 'Left':
                        ind_shift = 21 * 3
                    hand_points = hands[i].get('lmList')
                    for j in range(len(hand_points)):
                        for k in range(3):
                            points[ind_shift + j * 3 + k] = hand_points[j][k]

                # Recognize the pose and collect points
                img2 = self.pose_detector.findPose(frame)
                lmList, bboxInfo = self.pose_detector.findPosition(frame, bboxWithHands=False)
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

        return tensor, self.name_with_label[self.videos_names[index]][0], video_path


def read_preprocess_json(json_name, videos_root):
    """
    Read json and separate videos on predefined subsets(train, val, test). Check for existence of videos.

    :param json_name: path or name of json file in format {'video_name.mp4': {'subset': 'train', 'action': [class_num,
    start_frame, end_frame]}}
    :param videos_root: root folder of all videos
    :return: train, validation and test dictionaries in format {'video_name.mp4': [class_num, start_frame, end_frame]}
    """
    videos = json.load(open(json_name))
    train, val, test = dict(), dict(), dict()
    for name in os.listdir(videos_root):
        name = name[:-4]
        if videos.get(name) is None:
            continue
        if videos[name]['subset'] == 'train':
            train[name] = videos[name]['action']
        elif videos[name]['subset'] == 'val':
            val[name] = videos[name]['action']
        elif videos[name]['subset'] == 'test':
            test[name] = videos[name]['action']
    return train, val, test


def read_classes(path):
    classes = dict()
    with open(path, 'r') as file:
        for line in file:
            line = line.strip().split('\t')
            key = int(line[0])
            value = line[1]
            classes[key] = value
    return classes
