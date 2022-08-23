"""
Testing script for detecting driver drowsiness.
"""
import argparse

import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from torch.utils.data import DataLoader

from dataset import WLFWDatasets
from pfld import PFLDInference
from utils import plot_pose_cube
from calculate_eye_mouth_distance import EyeMouthDistance
from tqdm import tqdm

import matplotlib.pyplot as plt
import cv2
import os


def validate(wlfw_val_dataloader, plfd_backbone, args):
    plfd_backbone.eval()

    with torch.no_grad():

        index = 0
        drowsy = 0
        head = 0
        normal = 0
        eye_distances = []
        mouth_distances = []
        yaws, pitches, rolls = [], [], []
        for img in tqdm(wlfw_val_dataloader):
            # if index >= args.test_size:
            #     break
            img.requires_grad = False
            img = img.cuda(non_blocking=True)

            plfd_backbone = plfd_backbone.cuda()
            pose, landmarks = plfd_backbone(img)

            landms_tmp = landmarks.cpu().numpy()
            landms_tmp = landms_tmp.reshape(landms_tmp.shape[0], -1, 2)

            # show result
            show_img = np.array(np.transpose(img[0].cpu().numpy(), (1, 2, 0)))
            show_img = (show_img * 255).astype(np.uint8)
            np.clip(show_img, 0, 255)
            draw = show_img.copy()
            yaw = pose[0][0] * 180 / np.pi
            pitch = pose[0][1] * 180 / np.pi
            roll = pose[0][2] * 180 / np.pi
            pre_landmark = landms_tmp[0] * [112, 112]

            eye_mouth_distance = EyeMouthDistance(pre_landmark)

            eye_distances.append(eye_mouth_distance.eye_distance)
            mouth_distances.append(eye_mouth_distance.mouth_distance)

            y = yaw.cpu().numpy() + 40.0
            p = pitch.cpu().numpy() + 60.0
            r = roll.cpu().numpy() + 35.0
            # print(y, p, r)
            # print('\n')
            yaws.append(y)
            pitches.append(p)
            rolls.append(r)
            # for (x, y) in pre_landmark.astype(np.int8):
            #     cv2.circle(draw, (int(x), int(y)), 1, (0, 255, 0), 1)
            if (eye_mouth_distance.eye_distance < np.array(eye_distances).mean() * 0.4 or
                    eye_mouth_distance.mouth_distance > np.array(mouth_distances).mean() * 1.2):
                print(eye_mouth_distance.eye_distance, eye_mouth_distance.mouth_distance)
                cv2.imwrite('./results/drowsiness_dataset/test/closed_yawn/{}.png'.format(index + 0), draw)

                drowsy += 1
            elif (y < np.array(yaws).mean() * 0.7 or
                    y > np.array(yaws).mean() * 1.3 or
                    p < np.array(pitches).mean() * 0.7 or
                    p > np.array(pitches).mean() * 1.3 or
                    r < np.array(rolls).mean() * 0.7 or
                    r > np.array(rolls).mean() * 1.3):
                # draw = plot_pose_cube(draw, yaw, pitch, roll, size=draw.shape[0] // 2)
                cv2.imwrite('./results/drowsiness_dataset/test/head/{}.png'.format(index), draw)

                head += 1
            else:
                cv2.imwrite('./results/drowsiness_dataset/test/normal/{}.png'.format(index + 0), draw)

                normal += 1

            index += 1

        # fig1 = plt.figure()
        # plt.plot(eye_distances)
        # plt.savefig('./results/non_eye_distances.png')
        # print(np.array(eye_distances).mean(), np.array(eye_distances).mean() * 0.2)
        #
        # fig2 = plt.figure()
        # plt.plot(mouth_distances)
        # plt.savefig('./results/non_mouth_distances.png')
        # print(np.array(mouth_distances).mean(), np.array(mouth_distances).mean() * 1.2)

        # fig3 = plt.figure()
        # plt.plot(yaws)
        # plt.savefig('./results/yaw.png')
        # fig4 = plt.figure()
        # plt.plot(pitches)
        # plt.savefig('./results/pitch.png')
        # fig5 = plt.figure()
        # plt.plot(rolls)
        # plt.savefig('./results/roll.png')
        # print(np.array(yaws).mean(), np.array(pitches).mean(), np.array(rolls).mean())

        # accuracy = head / wlfw_val_dataloader.__len__()
        # print(drowsy)
        # print(head)
        # print(normal)

        return drowsy, head, normal


def main(args):
    checkpoint = torch.load(args.model_path)

    plfd_backbone = PFLDInference().cuda()

    plfd_backbone.load_state_dict(checkpoint)

    transform = transforms.Compose([transforms.ToTensor()])

    wlfw_val_dataset = WLFWDatasets(args.test_dataset, transform)
    wlfw_val_dataloader = DataLoader(
        wlfw_val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    drowsy = []
    head = []
    normal = []
    for i in range(10):
        d, h, n = validate(wlfw_val_dataloader, plfd_backbone, args)
        drowsy.append(d)
        head.append(h)
        normal.append(n)

        # print(drowsy, head)
        print(np.array(drowsy).mean(), np.array(head).mean(), np.array(normal).mean())


def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument('--model_path', default="./models/checkpoint_robust.pth", type=str)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--test_dataset',
                        default="/home/disk01/zgz/head_pose_estimation/data/drowsiness_dataset/test_data/????*/*.png",
                        type=str)
    parser.add_argument('--test_size', default=2500, type=int)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    args = parse_args()
    main(args)
