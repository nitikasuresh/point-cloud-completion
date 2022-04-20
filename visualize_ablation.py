#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse
import random
from torch.autograd import Variable

import torch
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms
import utils
from utils import PointLoss
from utils import distance_squre
import shapenet_part_loader
from D_net import D_net
from matplotlib import pyplot as plt
# from D_net_graph import D_net
from mayavi import mlab
# import open3d as o3d
from io_util import read_pcd, save_pcd
import numpy as np
from mayavi import mlab
import time

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot', default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int, default=0, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num', type=int, default=512, help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type=bool, default=False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG',
                    default='',
                    help="path to netG (to continue training)")
parser.add_argument('--infile', type=str, default='test_files/crop12.csv')
parser.add_argument('--infile_real', type=str, default='test_files/real11.csv')
parser.add_argument('--netD',
                    default='',
                    help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop', type=float, default=0.2)
parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
# Set the first parameter of '--point_scales_list' equal to (point_number + 512).
parser.add_argument('--point_scales_list', type=list, default=[2048, 512], help='number of points in each scales')
parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
parser.add_argument('--wtl2', type=float, default=0.9, help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
parser.add_argument('--cloud_size', type=int, default=1024, help='0 means do not use else use with this weight')

opt = parser.parse_args()
print(opt)


def plot_pcd(ax, pcd, color='Reds'):
	ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir='y', c=pcd[:, 0], s=0.5, cmap=color, vmin=-1, vmax=0.5)
	ax.set_axis_off()
	ax.set_xlim(-0.3, 0.3)
	ax.set_ylim(-0.3, 0.3)
	ax.set_zlim(-0.3, 0.3)


def distance_squre1(p1, p2):
	return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2


CHOICE = 'Car'

# '/media/tinglin/Francis_Louise/XIEBAO/ICRA21/Graph_PC_Completion_Public/Trained_Model/gen_net_Car2888.pth'
# gen_net_Car_Attention108
# gen_net_CarQ3196
# gen_net_CarQ2160

model_dic = {
	'Airplane': ''
}

id_dic = {
	'Airplane': 1,
	'Car': 2,
	'Chair': 1,
	'Laptop': 3,
	'Mug': 0,
	'Table': 0,
}

if CHOICE in ['Airplane', 'Laptop', 'Mug', 'Car']: #, 'Car'
	from Encoder1024 import Decoder
else:
	from Encoder import Decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_dset = shapenet_part_loader.PartDataset(root='./dataset/shapenetcore_partanno_segmentation_benchmark_v0/',
                                             classification=True, class_choice=CHOICE, npoints=opt.pnum, split='test')
test_loader = torch.utils.data.DataLoader(test_dset, batch_size=opt.batchSize,
                                          shuffle=False, num_workers=int(opt.workers))
gen_net = Decoder(opt.point_scales_list[0], opt.crop_point_num)
dis_net = D_net(opt.crop_point_num)
# dis_net = D_net(4, opt.crop_point_num)
USE_CUDA = True
criterion_PointLoss = PointLoss().to(device)


def weights_init_normal(m):
	classname = m.__class__.__name__
	if classname.find("Conv2d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("Conv1d") != -1:
		torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find("BatchNorm2d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)
	elif classname.find("BatchNorm1d") != -1:
		torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
		torch.nn.init.constant_(m.bias.data, 0.0)


# initialize generator and discriminator weights and device
if USE_CUDA:
	print("Using", torch.cuda.device_count(), "GPUs")
	gen_net = torch.nn.DataParallel(gen_net)
	gen_net.to(device)
	gen_net.apply(weights_init_normal)
# dis_net = torch.nn.DataParallel(dis_net)
# dis_net.to(device)
# dis_net.apply(weights_init_normal)

if opt.netG != '':
	gen_net.load_state_dict(torch.load(model_dic[CHOICE], map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netG)['epoch']
# print('G loaded')
# if opt.netD != '':
# 	dis_net.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
# 	resume_epoch = torch.load(opt.netD)['epoch']
# print('D loaded')
# print(resume_epoch)

print('total counts of test_loader: ', len(test_loader))

for i, data in enumerate(test_loader):
	
	if i != id_dic[CHOICE]: continue
	
	real_point, target, prior = data
	
	batch_size = real_point.size()[0]
	if batch_size < opt.batchSize: continue
	real_center = torch.FloatTensor(batch_size, 1, opt.crop_point_num, 3)
	input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
	input_cropped1 = input_cropped1.data.copy_(real_point)
	real_point = torch.unsqueeze(real_point, 1)
	input_cropped1 = torch.unsqueeze(input_cropped1, 1)  # input_cropped1.shape = [24, 1, 2024, 3]
	p_origin = [0, 0, 0]
	
	if opt.cropmethod == 'random_center':
		# set viewpoints
		choice = [torch.Tensor([1, 0, 0]), torch.Tensor([0, 0, 1]), torch.Tensor([1, 0, 1]),
		          torch.Tensor([-1, 0, 0]), torch.Tensor([-1, 1, 0])]
		for m in range(batch_size):
			# index = random.sample(choice, 1)  # Random choose one of the viewpoint
			
			if CHOICE == 'Airplane':
				index = [choice[0]]
			else:
				index = [choice[0]]
			
			# index = random.sample(choice, 1)
			distance_list1 = []
			p_center = index[0]
			for n in range(opt.pnum):
				distance_list1.append(distance_squre(real_point[m, 0, n], p_center))
			distance_order1 = sorted(enumerate(distance_list1), key=lambda x: x[1])
			#
			# distance_list2 = []
			# p_center = index[1]
			# for n in range(opt.pnum):
			# 	distance_list2.append(distance_squre(real_point[m, 0, n], p_center))
			# distance_order2 = sorted(enumerate(distance_list2), key=lambda x: x[1])
			
			for sp in range(opt.crop_point_num):
				input_cropped1.data[m, 0, distance_order1[sp][0]] = torch.FloatTensor([0, 0, 0])
				real_center.data[m, 0, sp] = real_point[m, 0, distance_order1[sp][0]]
			
			# for sp in range(opt.crop_point_num):
			# 	input_cropped1.data[m, 0, distance_order2[sp][0]] = torch.FloatTensor([0, 0, 0])
			# 	real_center.data[m, 0, sp] = real_point[m, 0, distance_order2[sp][0]]
	
	real_center = real_center.to(device)
	real_center = torch.squeeze(real_center, 1)
	input_cropped1 = input_cropped1.to(device)
	input_cropped1 = torch.squeeze(input_cropped1, 1)
	input_cropped1 = Variable(input_cropped1, requires_grad=False)
	
	# print(input_cropped1.shape) # (1, 2048, 3)
	# print(prior.shape)  # (1, 2048, 3)
	
	gen_net.eval()
	START = time.time()
	fake_center1, fake_fine, _, _, _, _ = gen_net(input_cropped1, prior)
	END = time.time()
	print('Time used: ', END - START)
	# fake_center1, fake_fine, _, _, _, _, _, _ = gen_net(input_cropped1, prior)
	
	# print(fake_fine.shape)
	
	CD_loss = criterion_PointLoss(torch.squeeze(fake_fine, 1), torch.squeeze(real_center, 1))
	
	fake_fine = fake_fine.cpu()
	fake_center1 = fake_center1.cpu()
	np_fake = fake_fine[0].detach().numpy()
	np_fake1 = fake_center1[0].detach().numpy()
	input_cropped1 = input_cropped1.cpu()
	np_crop = input_cropped1[0].numpy()
	
	real_point = torch.squeeze(real_point, 1)
	real_point = torch.squeeze(real_point, 0).numpy()
	# print(np_crop.shape, np_fake.shape, real_point.shape)
	
	x = np_crop[:, 0]  # x position of point
	y = np_crop[:, 1]  # y position of point
	z = np_crop[:, 2]  # z position of point
	
	xf = np_fake[:, 0]  # x position of point
	yf = np_fake[:, 1]  # y position of point
	zf = np_fake[:, 2]  # z position of point
	
	# d = np.sqrt(x ** 2 + y ** 2)  # Map Distance from sensor
	
	vals = 'height'
	if vals == "height":
		col = z
	else:
		col = d
	
	fig = mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
	mlab.points3d(x, y, z,
	              color=(0.9922, 0.9373, 0.9176),#(0.9922, 0.9373, 0.9176),
	              mode="sphere",
	              scale_factor=0.06,
	              figure=fig,
	              )
	
	mlab.points3d(xf, yf, zf,
	              color=(0.9999, 0.7137, 0.7569),#(0.9999, 0.7137, 0.7569),
	              mode="sphere",
	              scale_factor=0.06,
	              figure=fig,
	              )
	mlab.show()
	
	# x = real_point[:, 0]  # x position of point
	# y = real_point[:, 1]  # y position of point
	# z = real_point[:, 2]  # z position of point
	#
	# fig = mlab.figure(bgcolor=(1, 1, 1), size=(400, 400))
	# mlab.points3d(x, y, z,
	#               color=(0.9922, 0.9373, 0.9176),
	#               mode="sphere",
	#               scale_factor=0.06,
	#               figure=fig,
	#               )
	#
	# mlab.show()
	
	
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, projection='3d')
	# plot_pcd(ax, prior[0], color='Purples')
	# # ax.set_title('Output')
	# plt.show()
	
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, projection='3d')
	# plot_pcd(ax, np.concatenate((np_fake, np_crop), axis=0))
	# # ax.set_title('Output')
	# plt.show()

	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, projection='3d')
	# plot_pcd(ax, np_crop, color='Greys')
	# # ax.set_title('Input')
	# plt.show()
	#
	# fig = plt.figure(figsize=(4, 4))
	# ax = fig.add_subplot(111, projection='3d')
	# plot_pcd(ax, real_point, color='Blues')
	# # ax.set_title('Ground Truth')
	# plt.show()


