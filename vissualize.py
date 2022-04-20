#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import utils
from utils import *
import data_utils as d_utils
from dataset import *
# from Encoder import Decoder
from Encoder1024 import Decoder
from D_net import D_net
# import shapenet_part_loader
from mayavi import mlab

torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser()
#parser.add_argument('--dataset',  default='ModelNet40', help='ModelNet10|ModelNet40|ShapeNet')
parser.add_argument('--dataroot',  default='dataset/train', help='path to dataset')
parser.add_argument('--workers', type=int,default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--crop_point_num',type=int,default=1024,help='0 means do not use else use with this weight')
parser.add_argument('--nc', type=int, default=3)
parser.add_argument('--niter', type=int, default=300, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', type = bool, default = False, help='enables cuda')
parser.add_argument('--ngpu', type=int, default=2, help='number of GPUs to use')
parser.add_argument('--netG', default='Trained_Model/gen_net_Table_Attention22.pth', help="")
parser.add_argument('--netD', default='Trained_Model/dis_net_Table_Attention22.pth', help="")
parser.add_argument('--infile',type = str, default = 'test_files/crop12.csv')
parser.add_argument('--infile_real',type = str, default = 'test_files/real11.csv')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop',type=float,default=0.2)
parser.add_argument('--num_scales',type=int,default=3,help='number of scales')
# Set the first parameter of '--point_scales_list' equal to (point_number + 512).
parser.add_argument('--point_scales_list',type=list,default=[2048,1024],help='number of points in each scales')
parser.add_argument('--each_scales_size',type=int,default=1,help='each scales size')
parser.add_argument('--wtl2',type=float,default=0.9,help='0 means do not use else use with this weight')
parser.add_argument('--cropmethod', default = 'random_center', help = 'random|center|random_center')
parser.add_argument('--cloud_size',type=int,default=1024,help='0 means do not use else use with this weight')
opt = parser.parse_args()
print(opt)

def distance_squre1(p1,p2):
    return (p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

test_set = ShapeNet(train=False, npoints=opt.cloud_size, np_prior=opt.pnum)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.workers))

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
	dis_net = torch.nn.DataParallel(dis_net)
	dis_net.to(device)
	dis_net.apply(weights_init_normal)

if opt.netG != '':
	gen_net.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netG)['epoch']
	print('G loaded')
if opt.netD != '':
	dis_net.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
	resume_epoch = torch.load(opt.netD)['epoch']
# print('D loaded')
# print(resume_epoch)

for i, data in enumerate(test_loader):
	real_center, target, prior = data
	batch_size = real_center.size()[0]
	if batch_size < opt.batchSize: continue
	real_center = real_center.float()
	input_cropped1 = torch.FloatTensor(batch_size, opt.pnum, 3)
	input_cropped1 = input_cropped1.data.copy_(real_center)
	
	real_center = torch.unsqueeze(real_center, 1)
	input_cropped1 = torch.unsqueeze(input_cropped1, 1)
	real_center = real_center.to(device)
	target = target.to(device)  
	real_center = torch.squeeze(real_center, 1)
	input_cropped1 = input_cropped1.to(device)
	input_cropped1 = torch.squeeze(input_cropped1, 1)
	input_cropped1 = Variable(input_cropped1, requires_grad=False)
	
	gen_net.eval()
	fake_fine1, fake_fine, conv11, conv12 = gen_net(input_cropped1)
	# CD_loss = criterion_PointLoss(torch.squeeze(fake_fine, 1), torch.squeeze(real_center, 1))

	fake_fine = target.cpu()
	np_fake = fake_fine[0].detach().numpy()
	input_cropped1 = input_cropped1.cpu()
	np_crop = real_center[0].cpu().numpy()

	print(np_fake.shape, np_crop.shape, real_center.shape)
	# display = np.vstack((np_fake, np_fake1, np_crop))
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

	fig = mlab.figure(bgcolor=(1, 1, 1), size=(640, 500))
	mlab.points3d(x, y, z,
						 color=(0.9, 0.87, 0.54),
	                     mode="sphere",
	                     scale_factor = 0.01,
	                     figure=fig,
	                     )

	mlab.points3d(xf, yf, zf,
					 color=(0.54, 0.82, 0.94), #baby blue: 0.54, 0.82, 0.94
	                 mode="sphere",
	                 scale_factor = 0.01,
	                 figure=fig,
	                 )
	mlab.show()