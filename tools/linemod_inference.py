import argparse
from random import randint
import cv2
import numpy as np
import torch.utils.data
from torch.autograd import Variable
from datasets.linemod.linemod_cat_sim import PoseDataset as PoseDataset_linemod
from lib.knn.__init__ import KNearestNeighbor
from lib.network import PoseNet, PoseRefineNet
from lib.transformations import quaternion_matrix, quaternion_from_matrix
from tools.draw_cloud_point import convert3dpointto2d, convert_coordinates
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_root', type=str, default='', help='dataset root dir')
parser.add_argument('--model', type=str, default='', help='resume PoseNet model')
parser.add_argument('--refine_model', type=str, default='', help='resume PoseRefineNet model')
opt = parser.parse_args()

num_objects = 13
num_points = 500
refinement = True
iteration = 2
bs = 1


estimator = PoseNet(num_points=num_points, num_obj=num_objects)
estimator.cuda()
estimator.load_state_dict(torch.load(opt.model))
estimator.eval()

refiner = PoseRefineNet(num_points=num_points, num_obj=num_objects)
refiner.cuda()
refiner.load_state_dict(torch.load(opt.refine_model))
refiner.eval()

data = PoseDataset_linemod('eval', num_points, False, opt.dataset_root, 0.0, True)
points, choose, img, target, model_points, idx, original_image, original_model = data.get_item(randint(100, 100))

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


### REFINEMENT ###

if refinement:
    for ite in range(0, iteration):
        T = Variable(torch.from_numpy(my_t.astype(np.float32))) \
            .cuda().view(1, 3).repeat(num_points, 1).contiguous().view(1, num_points, 3)

        my_mat = quaternion_matrix(my_r)
        R = Variable(torch.from_numpy(my_mat[:3, :3].astype(np.float32))).cuda().view(1, 3, 3)
        my_mat[0:3, 3] = my_t

        new_points = torch.bmm((points - T), R).contiguous()
        pred_r, pred_t = refiner(new_points, emb, idx)
        pred_r = pred_r.view(1, 1, -1)
        pred_r = pred_r / (torch.norm(pred_r, dim=2).view(1, 1, 1))
        my_r_2 = pred_r.view(-1).cpu().data.numpy()
        my_t_2 = pred_t.view(-1).cpu().data.numpy()
        my_mat_2 = quaternion_matrix(my_r_2)
        my_mat_2[0:3, 3] = my_t_2

        my_mat_final = np.dot(my_mat, my_mat_2)
        my_r_final = copy.deepcopy(my_mat_final)
        my_r_final[0:3, 3] = 0
        my_r_final = quaternion_from_matrix(my_r_final, True)
        my_t_final = np.array([my_mat_final[0][3], my_mat_final[1][3], my_mat_final[2][3]])

        my_pred = np.append(my_r_final, my_t_final)
        my_r = my_r_final
        my_t = my_t_final


### DRAW MODEL PREDICTED ###

std_cloud_point = np.transpose(original_model)

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
