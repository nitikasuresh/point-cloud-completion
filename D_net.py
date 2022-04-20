import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor_1(nn.Module):
	def __init__(self, k, emb_dims, output_channels=40):
		super(FeatureExtractor_1, self).__init__()
		self.k = k
		self.emb_dims = emb_dims
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

		self.maxpool = nn.MaxPool2d((1, 2048), 1)

		self._attention1 = Attention(64) #64
		self._attention2 = Attention(64) #64
		self._attention3 = Attention(128) #128
		self._attention4 = Attention(256) #256

		self.fc1 = nn.Linear(448,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,16)
		self.fc4 = nn.Linear(16,1)
		self.bn_1 = nn.BatchNorm1d(256)
		self.bn_2 = nn.BatchNorm1d(128)
		self.bn_3 = nn.BatchNorm1d(16)
	def forward(self, x):
		batch_size = x.size(0)
		x = x.permute(0, 2, 1)
		x = get_graph_feature(x, k = self.k)
		x = self.conv1(x)
		x1 = x.max(dim = -1, keepdim = False)[0]
		x1 = self._attention1(x1)

		x = get_graph_feature(x1, k=self.k)
		x = self.conv2(x)
		x2 = x.max(dim=-1, keepdim=False)[0]
		x2 = self._attention2(x2)

		x = get_graph_feature(x2, k=self.k)
		x = self.conv3(x)
		x3 = x.max(dim=-1, keepdim=False)[0]
		x3 = self._attention3(x3)

		x = get_graph_feature(x3, k=self.k)
		x = self.conv4(x)
		x4 = x.max(dim=-1, keepdim=False)[0]
		x4 = self._attention4(x4)

		x2 = torch.squeeze(self.maxpool(x2), 2) 
		x3 = torch.squeeze(self.maxpool(x3), 2) 
		x4 = torch.squeeze(self.maxpool(x4), 2)
		
		Layers = [x4, x3, x2]
		x = torch.cat(Layers, 1)
		x = F.relu(self.bn_1(self.fc1(x)))
		x = F.relu(self.bn_2(self.fc2(x)))
		x = F.relu(self.bn_3(self.fc3(x)))
		x = self.fc4(x)
		return x

# num_center_point: number of generated center points from dense output
class D_net(nn.Module):
	def __init__(self, num_center_point):
		super(D_net, self).__init__()
		self.num_center_point = num_center_point
		self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
		self.conv2 = torch.nn.Conv2d(64, 64, 1)
		self.conv3 = torch.nn.Conv2d(64, 128, 1)
		self.conv4 = torch.nn.Conv2d(128, 256, 1)
		self.maxpool = torch.nn.MaxPool2d((self.num_center_point, 1), 1)
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(128)
		self.bn4 = nn.BatchNorm2d(256)
		self.fc1 = nn.Linear(448,256)
		self.fc2 = nn.Linear(256,128)
		self.fc3 = nn.Linear(128,16)
		self.fc4 = nn.Linear(16,1)
		self.bn_1 = nn.BatchNorm1d(256)
		self.bn_2 = nn.BatchNorm1d(128)
		self.bn_3 = nn.BatchNorm1d(16)

	def forward(self, x): # input size = [batch_size,1, num_center_point, 3]
		# print('discriminator input', x.shape)
		batch_size = x.size()[0]
		x = F.relu(self.bn1(self.conv1(x)))
		x_64 = F.relu(self.bn2(self.conv2(x)))
		x_128 = F.relu(self.bn3(self.conv3(x_64)))
		x_256 = F.relu(self.bn4(self.conv4(x_128)))

		# print('here-----------------')
		# print(x_64.shape)
		# print(x_128.shape)
		# print(x_256.shape)

		x_64 = torch.squeeze(self.maxpool(x_64))
		x_128 = torch.squeeze(self.maxpool(x_128))
		x_256 = torch.squeeze(self.maxpool(x_256))
		# size asserts
		# print(x_64.shape, x_128.shape, x_256.shape)
		if len(x_64.shape)==1: x_64 = x_64.view(batch_size, -1)
		if len(x_128.shape)==1: x_128 = x_128.view(batch_size, -1)
		if len(x_256.shape)==1: x_256 = x_256.view(batch_size,-1)

		Layers = [x_256, x_128, x_64]
		x = torch.cat(Layers, 1)
		x = F.relu(self.bn_1(self.fc1(x)))
		x = F.relu(self.bn_2(self.fc2(x)))
		x = F.relu(self.bn_3(self.fc3(x)))
		x = self.fc4(x)
		#print('discriminator output', x.shape)
		return x