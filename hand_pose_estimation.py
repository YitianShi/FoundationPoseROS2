import copy
from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from manotorch.manolayer import ManoLayer, MANOOutput

import open3d as o3d
from hamer.models import load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

from utils.pointcloud_alignment import PointCloudAlignment
import sam2

LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)

import trimesh
from vitpose_model import ViTPoseModel


class HandPoseEstimator:
    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, out_folder: str = 'demo_out', body_detector: str = 'vitdet'):
        print("Loading HaMer model...")
        self.model, self.model_cfg = load_hamer(checkpoint)
        # Setup HaMeR model
        # os.environ['CUDA_VISIBLE_DEVICES'] = "3"
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f"Using device: {self.device}")
        self.model = self.model.to(self.device)
        self.model.eval()

        # Load detector
        if body_detector == 'vitdet':
            from detectron2.config import LazyConfig
            import hamer
            cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
            detectron2_cfg = LazyConfig.load(str(cfg_path))
            detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
            for i in range(3):
                detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)
        elif body_detector == 'regnety':
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
            detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
            detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh = 0.4
            self.detector = DefaultPredictor_Lazy(detectron2_cfg)

        # keypoint detector
        self.cpm = ViTPoseModel(self.device)

        # Setup the renderer
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)

        # Make output directory if it does not exist
        self.out_folder = out_folder
        os.makedirs(out_folder, exist_ok=True)

        self.mano_layer = ManoLayer(
            mano_assets_root="_DATA/data/mano",
            use_pca=False,
            flat_hand_mean=True,
            n_comps=48
        ).to("cpu")

    def estimate_hand_pose(self, img_folder: str, side_view: bool = False, full_frame: bool = True, save_mesh: bool = False, rescale_factor: float = 2.0, file_type: list[str] = ['*.jpg', '*.png']):
        """
        Estimate hand pose from images and optionally align with a point cloud using ICP.

        Args:
            img_folder (str): Folder containing input images
            side_view (bool): Whether to render side view
            full_frame (bool): Whether to render full frame
            save_mesh (bool): Whether to save meshes
            rescale_factor (float): Factor for rescaling bounding boxes
            file_type (list[str]): List of file extensions to process
            point_cloud (np.ndarray, optional): Partial point cloud for ICP alignment
        """
        self.side_view = side_view
        self.full_frame = full_frame
        self.save_mesh = save_mesh

        # Get all demo images ends with .jpg or .png
        img_paths = [img for end in file_type for img in Path(img_folder).glob(end)]

        # Iterate over all images in folder
        for img_path in img_paths:
            img_cv2 = cv2.imread(str(img_path))

            # Detect humans in image
            det_out = self.detector(img_cv2)
            img = img_cv2.copy()[:, :, ::-1]

            det_instances = det_out['instances']
            valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
            pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
            pred_scores=det_instances.scores[valid_idx].cpu().numpy()

            # Detect human keypoints for each person
            vitposes_out = self.cpm.predict_pose(
                img,
                [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
            )

            bboxes = []
            is_right = []

            # Use hands based on hand keypoint detections
            for vitposes in vitposes_out:
                left_hand_keyp = vitposes['keypoints'][-42:-21]
                right_hand_keyp = vitposes['keypoints'][-21:]

                # Rejecting not confident detections
                keyp = left_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(0)
                keyp = right_hand_keyp
                valid = keyp[:,2] > 0.5
                if sum(valid) > 3:
                    bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                    bboxes.append(bbox)
                    is_right.append(1)

            if len(bboxes) == 0:
                continue

            boxes = np.stack(bboxes)
            right = np.stack(is_right)

            # Run reconstruction on all detected hands
            dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

            self.all_verts = []
            self.all_cam_t = []
            self.all_right = []

            for batch in dataloader:
                batch = recursive_to(batch, self.device)
                with torch.no_grad():
                    # Get HAMER predictions
                    self.hamer_out = self.model(batch)

                multiplier = (2*batch['right']-1)
                pred_cam = self.hamer_out['pred_cam']
                pred_cam[:,1] = multiplier*pred_cam[:,1]
                box_center = batch["box_center"].float()
                box_size = batch["box_size"].float()
                img_size = batch["img_size"].float()
                scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
                # scaled_focal_length = 748.52
                pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

                # Check if a point cloud is available for the current image
                pc_path = os.path.join(Path(img_folder).parent, 'point_cloud', Path(Path(img_path).stem.replace('image', 'point_cloud')).with_suffix('.ply'))

                # Convert to manotorch format
                verts = self.to_manotorch()
                self.hamer_out['pred_vertices'] = verts

                # Render the result
                n, img_fn = self.render_hand_pose(img_path, batch, pred_cam_t_full)

                # Apply ICP alignment if point cloud is provided
                if Path(pc_path).exists():
                    # Get predicted vertices
                    self.run_pointcloud_alignment(batch, pc_path)
                    # verts = self.align_hand_with_pointcloud(batch, pc_path, pred_cam_t_full)

            # Render front view
            if full_frame and len(self.all_verts) > 0:
                misc_args = dict(
                    mesh_base_color=LIGHT_BLUE,
                    scene_bg_color=(1, 1, 1),
                    focal_length=scaled_focal_length,
                )
                cam_view = self.renderer.render_rgba_multiple(self.all_verts, 
                                                              cam_t=self.all_cam_t, 
                                                              render_res=img_size[n], 
                                                              is_right=self.all_right, 
                                                              **misc_args)

                # Overlay image
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]

                cv2.imwrite(os.path.join(self.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])

    def estimate_hand_pose_single(self, img_cv2, pcd_img = None, 
                                  visualization_img = None,
                                  side_view: bool = False, 
                                  full_frame: bool = True, 
                                  save_mesh: bool = False, 
                                  rescale_factor: float = 2.0):
        """
        Estimate hand pose from images and optionally align with a point cloud using ICP.

        Args:
            img_folder (str): Folder containing input images
            side_view (bool): Whether to render side view
            full_frame (bool): Whether to render full frame
            save_mesh (bool): Whether to save meshes
            rescale_factor (float): Factor for rescaling bounding boxes
            point_cloud (np.ndarray, optional): Partial point cloud for ICP alignment
        """
        self.side_view = side_view
        self.full_frame = full_frame
        self.save_mesh = save_mesh

        # Detect humans in image
        det_out = self.detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = self.cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )

        bboxes = []
        is_right = []

        # Use hands based on hand keypoint detections
        for vitposes in vitposes_out:
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Rejecting not confident detections
            keyp = left_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(0)
            keyp = right_hand_keyp
            valid = keyp[:,2] > 0.5
            if sum(valid) > 3:
                bbox = [keyp[valid,0].min(), keyp[valid,1].min(), keyp[valid,0].max(), keyp[valid,1].max()]
                bboxes.append(bbox)
                is_right.append(1)

        if len(bboxes) == 0:
            print("No hands detected")
            return visualization_img

        boxes = np.stack(bboxes)
        right = np.stack(is_right)

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(self.model_cfg, img_cv2, boxes, right, rescale_factor=rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

        self.all_verts = []
        self.all_cam_t = []
        self.all_right = []

        for batch in dataloader:
            batch = recursive_to(batch, self.device)
            with torch.no_grad():
                # Get HAMER predictions
                self.hamer_out = self.model(batch)

            multiplier = (2*batch['right']-1)
            pred_cam = self.hamer_out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            # scaled_focal_length = 748.52
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().cpu().numpy()

            # Convert to manotorch format
            self.hamer_out['pred_vertices'] = self.to_manotorch()

            for n in range(batch['img'].shape[0]):
                # Get predicted vertices
                verts = self.hamer_out['pred_vertices'][n].detach().cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()

                # Add all verts and cams to list
                cam_t = pred_cam_t_full[n]
                verts[:,0] = (2*is_right-1)*verts[:,0]
                self.all_verts.append(verts)
                self.all_cam_t.append(cam_t)
                self.all_right.append(is_right)

        # Render front view
        if full_frame and len(self.all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            cam_view = self.renderer.render_rgba_multiple(self.all_verts, 
                                                            cam_t=self.all_cam_t, 
                                                            render_res=img_size[n], 
                                                            is_right=self.all_right, 
                                                            **misc_args)

            # Overlay image
            if visualization_img is None:
                visualization_img = img_cv2
            input_img = visualization_img.astype(np.float32)[:,:,::-1]/255.0
            input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
            input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
   
            # Save the point clouds to disk for debugging
            hand_pcds = o3d.geometry.PointCloud()
            for n, verts in enumerate(self.all_verts):
                # Create Open3D point clouds
                hand_pcd = o3d.geometry.PointCloud()
                hand_pcd.points = o3d.utility.Vector3dVector(verts)
                  # Red color for hand mesh
                trafo = np.eye(4)
                trafo[:3, 3] = self.all_cam_t[n]
                hand_pcd.transform(trafo)
                hand_pcds += hand_pcd
            hand_pcds.paint_uniform_color([1, 0, 0])
            
            # get hand point cloud via depth image
            pcd_hand_img = (cam_view[:,:,3:] >= 0.1).astype(int) * pcd_img
            pcd_hand = pcd_hand_img.reshape(-1, 3)
            # pcd_hand = pcd_hand[pcd_hand[:, 2] > 0 & pcd_hand[:, 2] < 1]
            pcd_hand_vis = o3d.geometry.PointCloud()
            pcd_hand_vis.points = o3d.utility.Vector3dVector(pcd_hand)
            pcd_hand_vis.paint_uniform_color([0, 1, 0])

            # get entire point cloud via depth image
            pcd_all = pcd_img.reshape(-1, 3)
            # pcd_img = pcd_img[pcd_img[:, 2] > 0 & pcd_img[:, 2] < 1]
            pcd_img_vis = o3d.geometry.PointCloud()
            pcd_img_vis.points = o3d.utility.Vector3dVector(pcd_all)
            pcd_img_vis.paint_uniform_color([0, 0, 1])

            # Save the point clouds to disk for debugging
            o3d.io.write_point_cloud(self.out_folder + "/hand.ply",
                                     pcd_hand_vis + hand_pcds + pcd_img_vis)

            # Save the point clouds to disk for debugging
            pl = ((pcd_img  - pcd_img.min()) / (pcd_img.max() - pcd_img.min()) * 255).astype(np.uint8)
            cv2.imwrite(os.path.join(self.out_folder, f'all.jpg'), pl)
            print("Hand pose estimation completed.")
            return (255*input_img_overlay).astype(np.uint8)[:, :, ::-1]

    def run_pointcloud_alignment(self, batch, pc_path):
        target = o3d.io.read_point_cloud(pc_path)
        target.paint_uniform_color([0, 1, 0])
        for n in range(batch['img'].shape[0]):
            if n == 0:
                continue
            verts = self.all_verts[n]
            # Create Open3D point clouds
            hand_pcd = o3d.geometry.PointCloud()
            hand_pcd.points = o3d.utility.Vector3dVector(verts)
            hand_pcd.paint_uniform_color([1, 0, 0])  # Red color for hand mesh
            trafo = np.eye(4)
            trafo[:3, 3] = self.all_cam_t[n]
            transformed_hand_pcd = copy.deepcopy(hand_pcd)
            transformed_hand_pcd.transform(trafo)
            o3d.visualization.draw_geometries([hand_pcd, target])
            o3d.io.write_point_cloud(f"original_scene_{n}.ply", target + transformed_hand_pcd)
            pca = PointCloudAlignment(transformed_hand_pcd, target, voxel_size=0.05)
            pca.run()


    def to_manotorch(self):
        """
        Convert the output of the model to manotorch format.
        Args:
            self.hamer_out (dict): Output of the model containing 'hand_pose' and 'betas'.
        Returns:
            verts (torch.Tensor): Vertices in manotorch format.
        """
        mano_params = self.hamer_out['pred_mano_params']
        batch_size = mano_params['hand_pose'].shape[0]

        global_orient = mano_params['global_orient']
        global_rot = [R.from_matrix(global_orient[i].detach().cpu().numpy()).as_rotvec() for i in range(batch_size)]
        hand_orient = mano_params['hand_pose']
        hand_rot = [R.from_matrix(hand_orient[i].detach().cpu().numpy()).as_rotvec() for i in range(batch_size)]
        pose = torch.from_numpy(np.concatenate([global_rot, hand_rot], axis=1)).reshape(batch_size, -1).float()

        shape = mano_params['betas'].detach().cpu().float()
        
        # The mano_layer's output contains:
        """
        MANOOutput = namedtuple(
            "MANOOutput",
            [
                "verts",
                "joints",
                "center_idx",
                "center_joint",
                "full_poses",
                "betas",
                "transforms_abs",
            ],
        )
        """
        mano_output: MANOOutput = self.mano_layer(pose, shape)
        verts = mano_output.verts # (B, 778, 3), root(center_joint) relative
        joints = mano_output.joints # (B, 21, 3), root relative
        transforms_abs = mano_output.transforms_abs  # (B, 16, 4, 4), root relative
        return verts

    def render_hand_pose(self, img_path, batch, pred_cam_t_full):
        batch_size = batch['img'].shape[0]
        for n in range(batch_size):
            # Get filename from path img_path
            img_fn, _ = os.path.splitext(os.path.basename(img_path))
            person_id = int(batch['personid'][n])
            white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
            input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
            input_patch = input_patch.permute(1,2,0).numpy()

            # Get predicted vertices
            verts = self.hamer_out['pred_vertices'][n].detach().cpu().numpy()
            is_right = batch['right'][n].cpu().numpy()
            # verts[:,0] = (2*is_right-1)*verts[:,0]

            # cam_t = pred_cam_t_full[n]

            regression_img = self.renderer(verts,
                                            self.hamer_out['pred_cam_t'][n].detach().cpu().numpy(),
                                            batch['img'][n],
                                            mesh_base_color=LIGHT_BLUE,
                                            scene_bg_color=(1, 1, 1),
                                            )

            if self.side_view:
                side_img = self.renderer(verts,
                                                self.hamer_out['pred_cam_t'][n].detach().cpu().numpy(),
                                                white_img,
                                                mesh_base_color=LIGHT_BLUE,
                                                scene_bg_color=(1, 1, 1),
                                                side_view=True)
                final_img = np.concatenate([input_patch, regression_img, side_img], axis=1)
            else:
                final_img = np.concatenate([input_patch, regression_img], axis=1)

            cv2.imwrite(os.path.join(self.out_folder, f'{img_fn}_{person_id}.png'), 255*final_img[:, :, ::-1])

            # Add all verts and cams to list
            cam_t = pred_cam_t_full[n]
            verts[:,0] = (2*is_right-1)*verts[:,0]
            self.all_verts.append(verts)
            self.all_cam_t.append(cam_t)
            self.all_right.append(is_right)

            # Save all meshes to disk
            if self.save_mesh:
                camera_translation = cam_t.copy()
                tmesh = self.renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                tmesh.export(os.path.join(self.out_folder, f'{img_fn}_{person_id}.obj'))
        return n,img_fn
    

    def skew_matrix(self, v):
        """Create skew-symmetric matrix from vector."""
        batch_size = v.shape[0]
        zero = torch.zeros(batch_size, 1, device=v.device)
        skew = torch.stack([
            torch.cat([zero, -v[:, 2:3], v[:, 1:2]], dim=1),
            torch.cat([v[:, 2:3], zero, -v[:, 0:1]], dim=1),
            torch.cat([-v[:, 1:2], v[:, 0:1], zero], dim=1)
        ], dim=1)
        return skew

    def align_hand_with_pointcloud(self, batch, point_cloud_path, pred_cam_t_full):
        """
        Align the predicted hand mesh with a partial point cloud using ICP.

        Args:
            hand_vertices (np.ndarray): Predicted hand mesh vertices of shape (N, 3)
            point_cloud (np.ndarray): Partial point cloud of shape (M, 3)
            max_iterations (int): Maximum number of ICP iterations
            convergence_threshold (float): Threshold for convergence

        Returns:
            np.ndarray: Aligned hand vertices of shape (N, 3)
        """
        for n in range(batch['img'].shape[0]):
            if n == 0:
                continue
            verts = self.hamer_out['pred_vertices'][n].detach().cpu().numpy()
            is_right = batch['right'][n].cpu().numpy()
            verts[:,0] = (2*is_right-1)*verts[:,0]
            # Create Open3D point clouds
            hand_pcd = o3d.geometry.PointCloud()
            hand_pcd.points = o3d.utility.Vector3dVector(verts)
            hand_pcd.paint_uniform_color([1, 0, 0])  # Red color for hand mesh
            # Transform the hand mesh to the camera coordinate system
            transformed_hand_pcd = copy.deepcopy(hand_pcd)
            trafo = np.eye(4)
            trafo[:3, 3] = np.array(self.hamer_out['pred_cam'].cpu()[n])[[1 ,2, 0]]/10
            transformed_hand_pcd.transform(trafo)
            point_cloud_pcd = o3d.io.read_point_cloud(point_cloud_path)
            point_cloud_pcd.paint_uniform_color([0, 1, 0])  # Green color for point cloud
            # Transform the point cloud to the robot base
            camera_pose_name = Path(point_cloud_path).stem.replace('point_cloud', 'image') + '.txt'
            camera_pose_path = os.path.join(Path(point_cloud_path).parent.parent, 'camera_poses', camera_pose_name)
            camera_pose = np.loadtxt(camera_pose_path)
            #point_cloud_pcd.transform(np.linalg.inv(camera_pose))
            # Compute normals for the point cloud
            point_cloud_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))
            hand_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.05, max_nn=30))

            # o3d.io.write_point_cloud(f"hand_{n}.ply", hand_pcd)
            # o3d.io.write_point_cloud(f"scene_{n}.ply", point_cloud_pcd)
            # self.create_trimesh_scene(hand_pcd, point_cloud_pcd)
            combined_pcd = hand_pcd + transformed_hand_pcd + point_cloud_pcd
            # Save the point clouds to disk for debugging
            o3d.io.write_point_cloud(f"full_scene_{n}.ply", combined_pcd)

            # Perform ICP alignment
            reg_p2p = o3d.pipelines.registration.registration_icp(
                hand_pcd,
                point_cloud_pcd,
                max_correspondence_distance=0.05,
                init=np.eye(4),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            )
            source_temp = copy.deepcopy(hand_pcd)
            target_temp = copy.deepcopy(point_cloud_pcd)
            source_temp.paint_uniform_color([1, 0.706, 0])
            target_temp.paint_uniform_color([0, 0.651, 0.929])
            source_temp.transform(reg_p2p.transformation)
            # Apply the transformation to the hand mesh
            aligned_scene = target_temp + source_temp
            o3d.io.write_point_cloud(f"aligned_scene_{n}.ply", aligned_scene)
            o3d.visualization.draw_geometries([source_temp, target_temp],
                                            zoom=0.4459,
                                            front=[0.9288, -0.2951, -0.2242],
                                            lookat=[1.6784, 2.0612, 1.4451],
                                            up=[-0.3402, -0.9189, -0.1996])

        return hand_vertices_aligned

    def create_trimesh_scene(self, hand_pcd, scene_pcd):
        # Create a scene
        scene = trimesh.Scene()
        scene.add_geometry(trimesh.creation.axis())

        record = trimesh.PointCloud(scene_pcd.points)
        scene.add_geometry(record)
        # hand = trimesh.PointCloud(hand_pcd.points)
        # scene.add_geometry(hand)

        scene.save_image(visible=False, offscreen=True)

def main():
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default="_DATA/hamer_ckpts/checkpoints/hamer.ckpt", help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='images_20250416_103900/rgb', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=True, help='If set, render side view also')
    parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=True, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='vitdet', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')

    args = parser.parse_args()
    checkpoint_path = os.path.join(os.path.dirname(__file__), args.checkpoint)
    img_folder = os.path.join(os.path.dirname(__file__), args.img_folder)

    pose_estimator = HandPoseEstimator(checkpoint_path, args.out_folder, args.body_detector)
    import time
    time_start = time.time()
    pose_estimator.estimate_hand_pose(img_folder, args.side_view, args.full_frame, args.save_mesh, args.rescale_factor, args.file_type)
    time_end = time.time()
    print(f"Time taken: {time_end - time_start} seconds")



if __name__ == '__main__':
    main()
