"""
Interface for utilsing detect face.
"""
import os
import torch
from itertools import product as product
import numpy as np
from torchvision import transforms as trans
from math import floor

import cv2


def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    ###loc[cx的预测，cy的预测，w相对于锚框w的log回归预测，h相对于锚框h的log回归预测]####
    ###priors[锚框cx,锚框cy，锚框w，锚框h]####
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    #####boxes   [cx,cy,w,h]--->[x1,y1,x2,y2]####
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (tensor): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    landms = torch.cat((priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
                        priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
                        ), dim=1)
    return landms


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class PriorBox(object):
    def __init__(self, image_size=None):
        super(PriorBox, self).__init__()
        self.min_sizes = [[16, 32], [64, 128], [256, 512]]
        self.steps = [8, 16, 32]
        self.clip = False
        self.image_size = image_size
        self.feature_maps = [[floor(self.image_size[0] / step), floor(self.image_size[1] / step)] for step in
                             self.steps]
        # self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        output = torch.Tensor(anchors).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


class detect_face():
    def __init__(self, model_path, im_height=1280, im_width=768):
        if not os.path.exists(model_path):
            print('model path is error')
        else:
            self.model = torch.load(model_path)
            # self.model_onnx = ort.InferenceSession(model_path2)
            # self.model_onnx.get_modelmeta()

            self.im_height = im_height
            self.im_width = im_width
            self.trans = trans.Compose([
                trans.ToTensor(),
                trans.Normalize([0.5], [0.5])
            ])
            self.scale = torch.Tensor([self.im_width, self.im_height, self.im_width, self.im_height])
            self.resize = 1
            self.device = torch.device("cpu")
            self.variance = [0.1, 0.2]
            self.confidence_threshold = 0.8
            self.nms_threshold = 0.3
            priorbox = PriorBox(image_size=(im_height, im_width))
            priors = priorbox.forward()
            priors = priors.to(self.device)
            self.prior_data = priors.data

    def detect(self, img):
        ratio = 1
        # img=cv2.resize(img, (self.im_width, self.im_height), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = self.trans(img)
        img = img.unsqueeze(0)

        loc, conf, landm = self.model(img)
        loc = loc[0]
        conf = conf[0]
        landm = landm[0]

        # loc, conf, landm = self.model_onnx.run(['output0', 'output1', 'output2'], {'input0': img.numpy()})
        # loc=torch.from_numpy(loc[0])
        # conf=torch.from_numpy(conf[0])
        # landm=torch.from_numpy(landm[0])
        boxes = decode(loc.squeeze(0).data, self.prior_data, self.variance)
        boxes = boxes * self.scale / self.resize
        boxes = boxes.cpu().numpy()
        scores = conf.data.cpu().numpy()[:, 1]
        landm = decode_landm(landm.squeeze(0).data, self.prior_data, self.variance)
        scale1 = torch.Tensor([self.im_width, self.im_height, self.im_width, self.im_height,
                               self.im_width, self.im_height, self.im_width, self.im_height,
                               self.im_width, self.im_height])
        scale1 = scale1.to(self.device)
        landm = landm * scale1 / self.resize
        landm = landm.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landm = landm[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:5000]
        boxes = boxes[order]
        landm = landm[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landm = landm[keep]

        # keep top-K faster NMS
        dets = dets[:750, :]
        landm = landm[:750, :]

        dets = np.concatenate((dets, landm), axis=1)
        dets = dets * ratio
        print(dets)
        return dets


if __name__ == '__main__':
    model_path1 = './models_file/facedetect_model.pth'
    face_detector = detect_face(model_path1)
    img = cv2.imread(r'D:\BCTC\spc_capture\img_allliv\ir_panorama_20210721-161942.958925.png')
    reslut = face_detector.detect(img)
    print(reslut)
