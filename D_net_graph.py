import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstraction, get_graph_feature

class D_net(nn.Module):
	def __init__(self, k, num_center_point):

		super(D_net, self).__init__()

		self.k = k
		self.num_center_point = num_center_point

		self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
								   nn.BatchNorm2d(64),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
								   nn.BatchNorm2d(64),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
								   nn.BatchNorm2d(128),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
								   nn.BatchNorm2d(256),
								   nn.LeakyReLU(negative_slope=0.2))

		self.maxpool = nn.MaxPool2d((1, num_center_point), 1)

		self.fc1 = nn.Linear(384, 128)
		self.fc2 = nn.Linear(128, 16)
		self.fc3 = nn.Linear(16, 1)
		self.bn_1 = nn.BatchNorm1d(128)
		self.bn_2 = nn.BatchNorm1d(16)
		# self.bn_3 = nn.BatchNorm1d(16)

	def forward(self, x):
		batch_size = x.size(0)
		
		x = x.squeeze()
		
		x = x.permute(0, 2, 1)
		x = get_graph_feature(x, k = self.k)
		x = self.conv1(x)
		x = x.max(dim = -1, keepdim = False)[0]

		x = get_graph_feature(x, k=self.k)
		x = self.conv2(x)
		x = x.max(dim=-1, keepdim=False)[0]

		x = get_graph_feature(x, k=self.k)
		x = self.conv3(x)
		x3 = x.max(dim=-1, keepdim=False)[0]

		x = get_graph_feature(x3, k=self.k)
		x = self.conv4(x)
		x = x.max(dim=-1, keepdim=False)[0]
		
		# print('----------------here')
		x3 = torch.squeeze(self.maxpool(x3), 2) # [8, 128]
		# print(x3.shape)
		x = torch.squeeze(self.maxpool(x), 2) # [8, 256]
		# print(x.shape)
		
		x = torch.cat((x3, x), dim=1) # [8, 384]
		# print(output.shape)
		# exit()
		
		# x = output.view(batch_size, -1, 1)

		x = F.relu(self.bn_1(self.fc1(x)))
		x = F.relu(self.bn_2(self.fc2(x)))
		# x = F.relu(self.bn_3(self.fc3(x)))
		x = self.fc3(x)
		# print('discriminator output', x.shape)
		return x
