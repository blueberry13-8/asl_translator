import time
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.PoseModule import PoseDetector

from dataset.asl_dataset import ASLDataset, read_classes, read_preprocess_json


def webcum():
    # Play with cv2 and cvzone
    cap = cv2.VideoCapture(0)
    hands_detector = HandDetector()
    pose_detector = PoseDetector()
    while True:
        success, img = cap.read()
        hands, img = hands_detector.findHands(img)
        img = pose_detector.findPose(img)
        lmList, bboxInfo = pose_detector.findPosition(img, bboxWithHands=False)
        cv2.imshow("CUM", img)
        cv2.waitKey(1)

        # print(lmList)
        # print(bboxInfo)
        # print()
        # time.sleep(2)


def load_dataset():
    train, val, test = read_preprocess_json('wlasl_dataset/nslt_100.json', 'wlasl_dataset/videos')
    classes = read_classes('wlasl_dataset/wlasl_class_list.txt')
    train_dataset = ASLDataset('wlasl_dataset/videos', train, classes)
    return train_dataset


if __name__ == '__main__':
    # webcum()
    dataset = load_dataset()
    print(dataset[0])
    # TODO: remove video's path from dataset return. It was added for debugging
