source /opt/ros/humble/setup.bash
conda activate foundationpose_ros
python FoundationPose/bundlesdf/run_nerf.py  --ref_view_dir "demo_data/images_20250503_115215"
