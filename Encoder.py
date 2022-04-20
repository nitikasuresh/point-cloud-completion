import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch
from pointnet_util import PointNetSetAbstraction, get_graph_feature
from Attention.Cluster_Attention import MultiHeadAttention as MHA
from Attention.positionwiseFeedForward import PositionwiseFeedForward

class Attention(nn.Module):

	def __init__(self, Fea, q=1, v=1, h=1, dropout = 0.3):
		super(Attention, self).__init__()
		# attention
		# eight heads for now
		self.q, self.v, self.h = q, v, h
		self.dropout = dropout
		# input should be (batch, cluster, feature)
		# (b, N, feature)
		self.Fea = Fea

		self.skipAttention = MHA(self.Fea, self.q, self.v, self.h)
		self.feedForward = PositionwiseFeedForward(self.Fea)
		self.layerNorm1 = nn.LayerNorm(self.Fea)
		self.layerNorm2 = nn.LayerNorm(self.Fea)
		self.dropout = nn.Dropout(p=self.dropout)

	def forward(self, x):
		# print('here-------------------------')
		# print(x.shape)
		x = x.permute(0, 2, 1)
		residual = x
		x = self.skipAttention(query = x, key = x, value = x)
		x = self.dropout(x)
		x = self.layerNorm1(x + residual)

		# Feed forward
		residual = x
		x = self.feedForward(x)
		x = self.dropout(x)
		x = self.layerNorm2(x + residual)
		x = x.permute(0, 2, 1)
		return x

class Encoder(nn.Module):
	def __init__(self, num_points):
		super(Encoder, self).__init__()
		self.fe1 = FeatureExtractor_1(4, 1024)
		self.fe2 = FeatureExtractor_2(4, 1024)
		self.out_layer = nn.MaxPool2d((1, 2), 1)

	def forward(self, x, prior):
		out_1, conv11, conv12 = self.fe1(x) # (batch_size, 512, 3) || (batch_size, 1920)
		out_2, conv21, conv22 = self.fe2(prior) # (batch_size, 1920)
		out = torch.cat((out_1, out_2), 2) # (batch_size, 1920, 2)
		# print(out.shape)
		out = self.out_layer(out).view(-1, 1920) # (batch_size, 1920)

		return out, conv11, conv12, conv21, conv22

	def downsampling(self, x): # (batch_size, 2048, 3)
		pass


class FeatureExtractor_1(nn.Module):
	def __init__(self, k, emb_dims, output_channels=40):
		super(FeatureExtractor_1, self).__init__()
		self.k = k
		self.emb_dims = emb_dims
		self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
								   nn.BatchNorm2d(32),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv2 = nn.Sequential(nn.Conv2d(32*2, 64, kernel_size=1, bias=False),
								   nn.BatchNorm2d(64),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
								   nn.BatchNorm2d(128),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
								   nn.BatchNorm2d(256),
								   nn.LeakyReLU(negative_slope=0.2))
		self.conv5 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(512),
		                           nn.LeakyReLU(negative_slope=0.2))
		self.conv6 = nn.Sequential(nn.Conv2d(512 * 2, self.emb_dims, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(self.emb_dims),
		                           nn.LeakyReLU(negative_slope=0.2))
		
		# self._attention1 = Attention(32) #64
		# self._attention2 = Attention(64)  # 64
		# self._attention3 = Attention(128)  # 128
		# self._attention4 = Attention(256)  # 256
		# self._attention5 = Attention(512)  # 128
		self._attention6 = Attention(1024)  # 256
		
		self.maxpool = nn.MaxPool2d((1, 2048), 1)


	def forward(self, x):
		batch_size = x.size(0)
		x = x.permute(0, 2, 1)
		x = get_graph_feature(x, k = self.k)
		x = self.conv1(x)
		x1 = x.max(dim = -1, keepdim = False)[0]
		# x1 = self._attention1(x1)

		x = get_graph_feature(x1, k=self.k)
		x = self.conv2(x)
		x2 = x.max(dim=-1, keepdim=False)[0]
		# x2 = self._attention2(x2)

		x = get_graph_feature(x2, k=self.k)
		x = self.conv3(x)
		x3 = x.max(dim=-1, keepdim=False)[0]
		# x3 = self._attention3(x3)

		x = get_graph_feature(x3, k=self.k)
		x = self.conv4(x)
		x4 = x.max(dim=-1, keepdim=False)[0]
		# x4 = self._attention4(x4)
		
		x = get_graph_feature(x4, k=self.k)
		x = self.conv5(x)
		x5 = x.max(dim=-1, keepdim=False)[0]
		# x5 = self._attention5(x5)
		
		x = get_graph_feature(x5, k=self.k)
		x = self.conv6(x)
		x6 = x.max(dim=-1, keepdim=False)[0]
		x6 = self._attention6(x6)
		
		x3 = torch.squeeze(self.maxpool(x3), 2)
		x4 = torch.squeeze(self.maxpool(x4), 2)
		x5 = torch.squeeze(self.maxpool(x5), 2)
		x6 = torch.squeeze(self.maxpool(x6), 2)
		#
		output = torch.cat((x3, x4, x5, x6), dim=1)
		output = output.view(batch_size, -1, 1)
		
		return output, x1, x2


class FeatureExtractor_2(nn.Module):
	def __init__(self, k, emb_dims, output_channels=40):
		super(FeatureExtractor_2, self).__init__()
		self.k = k
		self.emb_dims = emb_dims
		self.conv1 = nn.Sequential(nn.Conv2d(6, 32, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(32),
		                           nn.LeakyReLU(negative_slope=0.2))
		self.conv2 = nn.Sequential(nn.Conv2d(32 * 2, 64, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(64),
		                           nn.LeakyReLU(negative_slope=0.2))
		self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(128),
		                           nn.LeakyReLU(negative_slope=0.2))
		self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(256),
		                           nn.LeakyReLU(negative_slope=0.2))
		self.conv5 = nn.Sequential(nn.Conv2d(256 * 2, 512, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(512),
		                           nn.LeakyReLU(negative_slope=0.2))
		self.conv6 = nn.Sequential(nn.Conv2d(512 * 2, self.emb_dims, kernel_size=1, bias=False),
		                           nn.BatchNorm2d(self.emb_dims),
		                           nn.LeakyReLU(negative_slope=0.2))
		
		# self._attention1 = Attention(32)  # 64
		# self._attention2 = Attention(64)  # 64
		# self._attention3 = Attention(128)  # 128
		# self._attention4 = Attention(256)  # 256
		# self._attention5 = Attention(512)  # 128
		self._attention6 = Attention(1024)  # 256
		
		self.maxpool = nn.MaxPool2d((1, 2048), 1)
		


	def forward(self, x):
		batch_size = x.size(0)
		x = x.permute(0, 2, 1)
		x = get_graph_feature(x, k=self.k)
		x = self.conv1(x)
		x1 = x.max(dim=-1, keepdim=False)[0]
		# x1 = self._attention1(x1)
		
		x = get_graph_feature(x1, k=self.k)
		x = self.conv2(x)
		x2 = x.max(dim=-1, keepdim=False)[0]
		# x2 = self._attention2(x2)
		
		x = get_graph_feature(x2, k=self.k)
		x = self.conv3(x)
		x3 = x.max(dim=-1, keepdim=False)[0]
		# x3 = self._attention3(x3)
		
		x = get_graph_feature(x3, k=self.k)
		x = self.conv4(x)
		x4 = x.max(dim=-1, keepdim=False)[0]
		# x4 = self._attention4(x4)
		
		x = get_graph_feature(x4, k=self.k)
		x = self.conv5(x)
		x5 = x.max(dim=-1, keepdim=False)[0]
		# x5 = self._attention5(x5)
		
		x = get_graph_feature(x5, k=self.k)
		x = self.conv6(x)
		x6 = x.max(dim=-1, keepdim=False)[0]
		x6 = self._attention6(x6)
		
		x3 = torch.squeeze(self.maxpool(x3), 2)
		x4 = torch.squeeze(self.maxpool(x4), 2)
		x5 = torch.squeeze(self.maxpool(x5), 2)
		x6 = torch.squeeze(self.maxpool(x6), 2)
		
		output = torch.cat((x3, x4, x5, x6), dim=1)
		output = output.view(batch_size, -1, 1)
		return output, x1, x2


class Decoder(nn.Module):
	def __init__(self, num_points, crop_point_num):
		super(Decoder, self).__init__()
		self.crop_point_num = crop_point_num
		self.latentfeature = Encoder(num_points)
		self.fc1 = nn.Linear(1920, 1024)
		self.fc2 = nn.Linear(1024, 512)

		self.fc1_1 = nn.Linear(1024, 128 * self.crop_point_num)
		self.fc2_1 = nn.Linear(512, 64 * 128)

		self.conv1_1 = torch.nn.Conv1d(self.crop_point_num, 512, 1)
		self.conv1_2 = torch.nn.Conv1d(512, 256, 1)
		self.conv1_3 = torch.nn.Conv1d(256, int((self.crop_point_num * 3) / 128), 1)
		
		self.conv2_1 = torch.nn.Conv1d(128, 6, 1)

	def forward(self, x, prior): 

		x, conv11, conv12, conv21, conv22 = self.latentfeature(x, prior)

		x = F.relu(self.fc1(x))  # 1024
		x_2 = F.relu(self.fc2(x))  # 512

		x_2 = self.fc2_1(x_2)
		x_2 = x_2.reshape(-1, 128, 64)
		x_2 = self.conv2_1(x_2)

		x = F.relu(self.fc1_1(x))
		x = x.reshape(-1, self.crop_point_num, 128)
		x = F.relu(self.conv1_1(x))
		x = F.relu(self.conv1_2(x))
		x = self.conv1_3(x)  # 12x128
		x = x.reshape(-1, 128, int(self.crop_point_num/128), 3)

		x_2 = x_2.reshape(-1, 128, 1, 3)
		
		# print(x.shape) #(6, 128, 4, 3)
		# print(x_2.shape) #(6, 128, 1, 3)
		
		x = x + x_2 # 128x4x3
		x = x.reshape(-1, self.crop_point_num, 3) # 512x3 Local Points

		return x_2.squeeze(), x, conv11, conv12, conv21, conv22


