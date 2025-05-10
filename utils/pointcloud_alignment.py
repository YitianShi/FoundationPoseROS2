import open3d as o3d
from open3d.geometry import PointCloud
from open3d.pipelines.registration import ICPConvergenceCriteria
import numpy as np
import copy

class PointCloudAlignment:
    def __init__(self, source: PointCloud, target: PointCloud, initial_transformation: np.ndarray = np.eye(4), voxel_size: float = 0.05):
        ''' Initialize the PointCloudAlignment class with source and target point clouds. '''
        self.source = source
        self.target = target
        self.voxel_size = voxel_size

        self.target = self.remove_table_ransac(self.target)

        self.source, self.target, self.source_down, self.target_down, self.source_fpfh, self.target_fpfh = self.prepare_dataset(
            voxel_size, initial_transformation)

    def remove_table_ransac(self, pcd, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
        """Remove the largest plane (e.g. table top) from the point cloud using RANSAC."""
        plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                                 ransac_n=ransac_n,
                                                 num_iterations=num_iterations)
        print(f"F Ebene: {plane_model}, entfernt {len(inliers)} Punkte (Tischplatte)")
        pcd_ohne_tisch = pcd.select_by_index(inliers, invert=True)
        return pcd_ohne_tisch

    def run(self):
        result_ransac = self.execute_global_registration(self.source_down, self.target_down,
                                                    self.source_fpfh, self.target_fpfh,
                                                    self.voxel_size)
        print(result_ransac)
        source_global_icp = copy.deepcopy(self.source)
        source_global_icp.transform(result_ransac.transformation)
        source_global_icp.paint_uniform_color([0, 0, 1])

        result_icp = self.refine_registration(self.source, self.target, self.source_fpfh, self.target_fpfh,
                                    self.voxel_size, result_ransac)
        print(result_icp)
        self.source.transform(result_icp.transformation)
        combined = self.source + self.target # + source_global_icp
        o3d.io.write_point_cloud("result_scene.ply", combined)

    def preprocess_point_cloud(self, pcd: PointCloud, voxel_size: float):
        print(":: Downsample with a voxel size %.3f." % voxel_size)
        pcd_down = pcd.voxel_down_sample(voxel_size)

        radius_normal = voxel_size * 2
        print(":: Estimate normal with search radius %.3f." % radius_normal)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

        radius_feature = voxel_size * 5
        print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down,
            o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
        return pcd_down, pcd_fpfh

    def prepare_dataset(self, voxel_size: float, initial_transformation: np.ndarray):
        print(":: Load two point clouds and disturb initial pose.")

        source = self.source # o3d.io.read_point_cloud('hand_1.ply')
        target = self.target # o3d.io.read_point_cloud('scene_1.ply')
        source.transform(initial_transformation)

        source_down, source_fpfh = self.preprocess_point_cloud(source, voxel_size)
        target_down, target_fpfh = self.preprocess_point_cloud(target, voxel_size)
        return source, target, source_down, target_down, source_fpfh, target_fpfh

    def execute_global_registration(self, source_down, target_down, source_fpfh,
                                    target_fpfh, voxel_size):
        distance_threshold = voxel_size * 1.5
        print(":: RANSAC registration on downsampled point clouds.")
        print("   Since the downsampling voxel size is %.3f," % voxel_size)
        print("   we use a liberal distance threshold %.3f." % distance_threshold)
        ransac_n = 500000
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh, True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3, [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                    0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold)
            ], o3d.pipelines.registration.RANSACConvergenceCriteria(ransac_n, 0.999))
        return result

    def execute_fast_global_registration(self, source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size):
        distance_threshold = voxel_size * 0.5
        print(":: Apply fast global registration with distance threshold %.3f" \
                % distance_threshold)
        result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down, target_down, source_fpfh, target_fpfh,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=distance_threshold))
        return result

    def refine_registration(self, source, target, source_fpfh, target_fpfh, voxel_size, result_ransac):
        distance_threshold = 200 # voxel_size * 0.4
        print(":: Point-to-plane ICP registration is applied on original point")
        print("   clouds to refine the alignment. This time we use a strict")
        print("   distance threshold %.3f." % distance_threshold)
        result = o3d.pipelines.registration.registration_icp(
            source, target, distance_threshold, np.eye(4), # result_ransac.transformation, 
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            ICPConvergenceCriteria(relative_fitness=1.000000e-06, relative_rmse=1.000000e-06, max_iteration=3000))
        return result

if __name__ == "__main__":
    # Load point clouds
    source = o3d.io.read_point_cloud("hand_1.ply")
    target = o3d.io.read_point_cloud("scene_1.ply")

    # Initialize the PointCloudAlignment class
    alignment = PointCloudAlignment(source, target)

    # Run the registration
    alignment.run()
