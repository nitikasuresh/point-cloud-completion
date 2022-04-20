import open3d as o3d
import numpy as np

pcd = o3d.io.read_point_cloud('/home/nsuresh/Documents/saga-net/SagaNet_exploration-main/data/unreal-data/train_ue_20220216/0/pcd_try_new/' + str(16)+ '.pcd')
pcd1 = o3d.io.read_point_cloud('/home/nsuresh/Documents/saga-net/SagaNet_exploration-main/data/complete/construction/SM_AsphaltRoller.pcd')
pcd_arr = np.asarray(pcd.points)
pcd1_arr = np.asarray(pcd1.points)
print(np.max(pcd_arr))
print(np.max(pcd1_arr))