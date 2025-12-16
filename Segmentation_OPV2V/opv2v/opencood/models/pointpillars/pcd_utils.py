# this can be removed or merged with the one in utils folder later
import os 
import numpy as np 
import open3d as o3d
from open3d.geometry import PointCloud

# returns pcd as numpy array
def read_points(file_path, dim = 4):
    suffix = os.path.splitext(file_path)[1] 
    assert suffix in ['.bin', '.ply', '.pcd']
    if suffix == '.bin':
        return np.fromfile(file_path, dtype=np.float32).reshape(-1, dim)
    elif suffix == '.pcd':
        # reads the point cloud with open3d
    	pcd = o3d.io.read_point_cloud(file_path)

        # returns the xyz data as np array
    	xyz = np.asarray(pcd.points, dtype=np.float32)

        # takes the pcd colors rgb values and compressed into one dim
    	intensity = np.expand_dims(np.asarray(pcd.colors)[:, 0], -1)

        # stacks the xyz and compressed rgb together
    	pcd_np = np.hstack((xyz, intensity), dtype=np.float32)
    	return pcd_np
    else:
        raise NotImplementedError

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
        data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
        point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 