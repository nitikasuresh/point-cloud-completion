import numpy as np
import open3d as o3d
import os

choice = '04225987'

dir = '/media/tinglin/Francis_Louise/XIEBAO/ICRA21/Graph_PC_Completion_Public' \
      '/dataset/shapenetcore_partanno_segmentation_benchmark_v0/' + choice + '/points/'
save_dir = '/media/tinglin/Francis_Louise/XIEBAO/ICRA21/Graph_PC_Completion_Public/data/pcd/'
counter = 0

# pp = o3d.io.read_point_cloud('/media/tinglin/Francis_Louise/XIEBAO/ICRA21/Graph_PC_Completion_Public/data/compare_lib/02691156_1af4b32eafffb0f7ee60c37cbf99c1c.pcd')
# # pp = np.asarray(pp)
# print(np.asarray(pp.points).shape)
# exit()

onlyfiles = [f for f in os.listdir(dir)]
all_items = len(onlyfiles)

for item in os.listdir(dir):
      all_items -= 1
      print(all_items)
      print(item)
      
      pcd = o3d.geometry.PointCloud()
      cloud = np.loadtxt(dir + item)  # Read the point cloud
      # cloud = np.asarray(cloud)
      # print(cloud.shape)
      #
      # pp = o3d.io.read_point_cloud(save_dir + choice + '_' + item[:-4] + '.pcd')
      # print(pp)
      # exit()
      
      pcd.points = o3d.utility.Vector3dVector(cloud)
      o3d.visualization.draw_geometries([pcd])  # Visualize the point cloud
      
      x = input('Save? y/n')
      
      if x == 'y':
            counter += 1
            o3d.io.write_point_cloud(save_dir + choice + '_' + item[:-4] + '.pcd', pcd)
            print(counter)