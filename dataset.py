import open3d as o3d
import torch
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import io
import sys
import random
#from utils import *
import contextlib
from inquary import *
# from dbscan import *

def resample_pcd(pcd, n):
	"""Drop or duplicate points so that pcd has exactly n points"""
	idx = np.random.permutation(pcd.shape[0])
	if idx.shape[0] < n:
		idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
	return pcd[idx[:n]]

def read_pcd(filename):
	pcd = o3d.io.read_point_cloud(filename, format = 'pcd')
	#o3d.visualization.draw_geometries([pcd])
	#print('read')
	return torch.from_numpy(np.array(pcd.points)).float()

class ShapeNet(data.Dataset): 
	def __init__(self, train = True, npoints = 2048, np_prior = 512):
		if train:
			self.list_path = 'data/train.list'
		else:
			self.list_path = 'data/test.list'
		self.npoints = npoints
		self.np_prior = np_prior
		self.train = train
		with open(os.path.join(self.list_path)) as file:
			self.model_list = [line.strip().replace('/', '/') for line in file]
			# print(self.model_list)
		random.shuffle(self.model_list)
		self.len = len(self.model_list * 100)
		# print(self.len)
		self.prior_set = get_prior_set('data/compare_lib/')
		#print('prior set len', len(self.prior_set))

	def __getitem__(self, index):
		model_id = self.model_list[index // 100]
		scan_id = index % 100

		if self.train:
			partial = read_pcd(os.path.join("data/train/pcd/",model_id.split('/', 2)[0]+'/'+ model_id.split('/', 2)[1]+ '/%d.pcd' % scan_id))
		else:
			partial = read_pcd(os.path.join("data/unreal-data/pcd", model_id.split('/', 2)[0]+'/'+ model_id.split('/', 2)[1]+ '/%d.pcd' % scan_id))
		complete = read_pcd(os.path.join("data/complete/", '%s.pcd' % model_id)).T
	

		complete *= 10
		# dellist = [j for j in range(0, len(complete))]
		# dellist = random.sample(dellist, len(complete) - 1000)
		# complete = np.delete(complete, dellist, axis=0)

		# partial = partial/10

		pose = np.loadtxt(os.path.join("data/unreal-data/pose/",model_id.split('/', 2)[0]+'/'+ model_id.split('/', 2)[1]+ '/%d.txt' % scan_id), delimiter = ' ')
		cam_ned_wrt_obj_ned = pose
		
		obj_ned_wrt_obj = np.array([[1, 0, 0, 0],[0, 0, -1, 0],[0, 1, 0, 0],[ 0, 0, 0, 1]])
		cam_real_wrt_cam_ned = np.array([[0, 0, 1, 0],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 0, 1]])
		cam_wrt_obj =  obj_ned_wrt_obj @ cam_ned_wrt_obj_ned @ cam_real_wrt_cam_ned 
		obj_wrt_cam = np.linalg.inv(cam_wrt_obj)
	
		truth_r = obj_wrt_cam[:3,:3]
		truth_t = obj_wrt_cam[:3,3]
		complete = np.dot(complete.T, truth_r.T) + truth_t

		# complete = complete.T

		#  inv_P = np.linalg.inv(pose)
		# print(target.shape)
		# complete = np.dot(obj_wrt_cam, np.concatenate([complete, np.ones((1, complete.shape[1]))], 0)).T[:, :3] #inv_P
		partial = resample_pcd(partial, 2048)#.cpu().detach().numpy()
		prior, idx = get_prior(partial, self.prior_set)
		# print('here-----------------------------------')
		# print(prior.dtype)
		# par_prior = np.concatenate((partial, prior),axis = 0)

		# obtain clusters with dbscan [cluster #, 32, 3]
		# eps, min_points = 0.02, 10
		# clusters = dbscan(partial, eps, min_points)
		# print('prior idx', idx)
		return partial/10, resample_pcd(complete/10, self.npoints), resample_pcd(prior/10, self.np_prior)

	def __len__(self):
		return self.len

if __name__ == '__main__':
	k = ShapeNet()
	
