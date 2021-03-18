# Copyright 2021 Morning Project Samurai, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR
# A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
# OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


import numpy as np
from PIL import Image, ImageChops
import torch
import torch.nn as nn
import torchvision
import torchvision.models as models


class Head(nn.Module):
    def __init__(self, num_classes=1000):
        super(Head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(num_classes, num_classes, bias=True)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(-1, 1000)
        x = self.fc(x)
        return x


class Features(nn.Module):
    def __init__(self, num_classes=1000):
        super(Features, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(resnet.conv1,
                                      resnet.bn1,
                                      resnet.relu,
                                      resnet.maxpool,
                                      resnet.layer1,
                                      resnet.layer2,
                                      resnet.layer3,
                                      resnet.layer4,
                                      nn.Conv2d(512, num_classes, 1))

    def forward(self, x):
        return self.features(x)


def predict(model, image):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225])
    ])

    x = transform(image).unsqueeze(0)
    y = torch.nn.functional.softmax(model(x), dim=-1)[0]
    feature_map = model[0](x)[0]
    weight = model[1].fc.weight

    return y, feature_map, weight


def create_cam(feature_map, weight, y, y_th, num_classes, target_classes):
    activation_map = torch.zeros(feature_map.shape[1:])
    for c in target_classes:
        for i in range(num_classes):
            if y[c] < y_th:
                continue
            activation_map += weight[c, i] * feature_map[i]
    return torch.clip(activation_map * 255. / torch.max(activation_map), 0)


def draw_cam(image, cam, out_path):
    im_cam = Image.fromarray(np.uint8(cam.detach().numpy())) \
        .resize(x.size) \
        .convert('RGB')
    im_cam = im_cam.resize(x.size).convert('RGB')
    im_cam = ImageChops.multiply(image, im_cam)
    im_cam.save(out_path)


if __name__ =='__main__':
    MODEL_PATH = '../models/epoch10.pth'
    person_classes = [124, 235, 236, 245, 316, 322, 380, 423,
                      516, 651, 699, 748, 901, 908, 928, 930]

    model = nn.Sequential(Features(), Head())
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    x = Image.open('../images/2.jpg').convert('RGB')
    y, feature_map, weight = predict(model, x)
    cam = create_cam(feature_map, weight, y, 1e-9, 1000, person_classes)
    draw_cam(x, cam, '../outputs/cam.png')
