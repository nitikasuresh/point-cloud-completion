from dataset import *
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import open3d as o3d
import torch

def resample_pcd(pcd, n):
	"""Drop or duplicate points so that pcd has exactly n points"""
	idx = np.random.permutation(pcd.shape[0])
	if idx.shape[0] < n:
		idx = np.concatenate([idx, np.random.randint(pcd.shape[0], size = n - pcd.shape[0])])
	return pcd[idx[:n]]

def get_prior_set(complete_dir):
	complete_pcds = []
	for f in os.listdir(complete_dir):
		complete_pcd = read_pcd(os.path.join(complete_dir+f))
		complete_pcd = resample_pcd(complete_pcd, 1024)
		complete_pcds.append(complete_pcd)
	#partial_pcds = [torch.from_numpy(partial_pcd1), torch.from_numpy(partial_pcd2), torch.from_numpy(partial_pcd3)]
	return complete_pcds

def read_pcd(filename):
	pcd = o3d.io.read_point_cloud(filename, format = 'pcd')
	return np.array(pcd.points)

def get_prior(p_cloud, c_clouds):
	#criterion_PointLoss = PointLoss()
	cd_loss = 1000
	partial_pcd = p_cloud
	fin_pcd = p_cloud
	for idx in range(len(c_clouds)):
		cd_loss_tmp = directed_hausdorff(partial_pcd, c_clouds[idx])[0]
		if cd_loss_tmp < cd_loss:
			cd_loss = cd_loss_tmp
			fin_pcd = c_clouds[idx]
			fin_idx = idx
	return fin_pcd.astype(np.float32), fin_idx

def get_batch_from_prior(p_clouds, c_clouds):
	fin_pcds = []
	for i in range(len(p_clouds)):
		prior = get_prior(p_clouds[i], c_clouds)
		fin_pcds.append(np.concatenate((p_clouds[i], prior),axis = 0))
		fin_pcds.append(get_prior(p_clouds[i], c_clouds))
	return torch.from_numpy(np.asarray(fin_pcds))