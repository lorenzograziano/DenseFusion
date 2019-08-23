import argparse
from random import randint

import cv2
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.linemod_cat_sim import PoseDataset as PoseDataset_linemod
from lib.knn.__init__ import KNearestNeighbor
from lib.network import PoseNet, PoseRefineNet
from tools.draw_cloud_point import convert3dpointto2d, convert_coordinates

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
parser.add_argument('--model', type=str, default='', help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default='', help='resume PoseRefineNet model')
opt = parser.parse_args()

num_objects = 13
objlist = [1, 2, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15]
num_points = 500
iteration = 2
bs = 1
dataset_config_dir = '/home/lorenzo/PycharmProjects/DenseFusionNew/datasets/linemod/dataset_config'
output_result_dir = '/home/lorenzo/PycharmProjects/DenseFusionNew/experiments/eval_result/linemod'
knn = KNearestNeighbor(1)

estimator = PoseNet(num_points=num_points, num_obj=num_objects)
estimator.cuda()
refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
refiner.cuda()
estimator.load_state_dict(torch.load(opt.model))
refiner.load_state_dict(torch.load(opt.refine_model))
estimator.eval()
refiner.eval()

data = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
points, choose, img, target, model_points, idx, original_image, original_model = data.get_item(randint(0, 1000))

# print(points.size())
# print(choose.size())
# print(img.size())
# print(target.size())
# print(model_points.size())
# print(idx.size())

points, choose, img, target, model_points, idx = Variable(points).cuda(), \
                                                 Variable(choose).cuda(), \
                                                 Variable(img).cuda(), \
                                                 Variable(target).cuda(), \
                                                 Variable(model_points).cuda(), \
                                                 Variable(idx).cuda()

pred_r, pred_t, pred_c, emb = estimator(img, points, choose, idx)

# print(pred_r.size())
# print(pred_t.size())
# print(pred_c.size())

pred_r = pred_r / torch.norm(pred_r, dim=2).view(1, num_points, 1)
pred_c = pred_c.view(bs, num_points)
how_max, which_max = torch.max(pred_c, 1)
pred_t = pred_t.view(bs * num_points, 1, 3)

my_r = pred_r[0][which_max[0]].view(-1).cpu().data.numpy()
my_t = (points.view(bs * num_points, 1, 3) + pred_t)[which_max[0]].view(-1).cpu().data.numpy()
my_pred = np.append(my_r, my_t)

print("rotation vector", my_r)
print("translation vector", my_t)

### DRAW MODEL PREDICTED ###
std_cloud_point = np.transpose(original_model)#[[1, 2, 0], :]

obj_coord = (convert_coordinates(my_pred, std_cloud_point))
obj_coord2d = convert3dpointto2d(obj_coord)

original_image = cv2.cvtColor(np.array(original_image), cv2.COLOR_BGR2RGB)
n, m = np.shape(obj_coord2d)

for i in range(m):
    center_x, center_y = obj_coord2d[:, i]
    cv2.circle(original_image, (int(center_x), int(center_y)), 1, (0, 0, 255), -1)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image', original_image)
key = cv2.waitKey(10000)  # pauses for 3 seconds before fetching next image
