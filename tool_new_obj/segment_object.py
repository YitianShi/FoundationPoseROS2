import torch
import glob
import os
import cv2
from sam2.sam2_video_predictor import SAM2VideoPredictor
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image 

data_name = "images_20250503_115215"
sample_step = 10

# === Step 1: Convert PNGs to video ===
input_dir = f"demo_data/{data_name}/rgb"
model_name = "facebook/sam2-hiera-large"
output_video = f'{input_dir}/output_rgb_video.mp4'
fps = 30

jpg_files = sorted(glob.glob(os.path.join(input_dir, '*.jpg')))
jpg_files = jpg_files[::sample_step]  # Sample every n frame
frame_names = [filename.split('.j')[0].replace('rgb', 'mask') for filename in jpg_files]
assert jpg_files, "No jpg files found!"
print(f"[INFO] Found {len(jpg_files)} jpg files in {input_dir}")

# check whether the mask folder exists
mask_dir = input_dir.replace('rgb', 'mask')
if not os.path.exists(mask_dir):
    os.makedirs(mask_dir)

# === MATPLOTLIB CLICK HANDLER ===
# Get frame dimensions
frame = cv2.imread(jpg_files[0])
height, width, _ = frame.shape

clicked_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x, y = int(event.xdata), int(event.ydata)
        clicked_points.append([x, y])
        print(f"[CLICK] ({x}, {y})")
        plt.scatter(x, y, c='red', s=40)
        plt.draw()

fig, ax = plt.subplots()
ax.imshow(frame)
plt.title("Click to add foreground points, then close window")
cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

# Write PNGs to video
if not os.path.exists(output_video):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    for fname in jpg_files:
        img = cv2.imread(fname)
        out.write(img)
    out.release()
    print(f"[INFO] Video saved to: {output_video}")
    del out

prompt_frame = 0  # Frame index to prompt on

# === LOAD MODEL ===
print("[INFO] Loading SAM2 model...")
predictor = SAM2VideoPredictor.from_pretrained(model_name).to("cuda")
state = predictor.init_state(output_video)

# After clicking, this holds your point prompts
point_coords = clicked_points
point_labels = [1] * len(clicked_points)

print("[INFO] Running SAM2 on prompt frame...")
frame_idx, object_ids, masks = predictor.add_new_points_or_box(
    state,
    obj_id=1,
    frame_idx=prompt_frame,
    points=point_coords,
    labels=point_labels
)

print(f"[RESULT] Frame {frame_idx}: {len(object_ids)} objects")

def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# === PROPAGATE MASKS ===
print("[INFO] Propagating masks across video...")
video_segments = {}  # video_segments contains the per-frame segmentation results
for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(state):
    video_segments[out_frame_idx] = {
        out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
        for i, out_obj_id in enumerate(out_obj_ids)
    }
    out_mask = out_mask_logits[0].cpu().numpy()
    out_mask = (out_mask > 0.0).astype(np.uint8)* 255
    out_mask = cv2.cvtColor(out_mask.squeeze(), cv2.COLOR_GRAY2BGR)
    # save the mask to a file
    plt.imsave(frame_names[out_frame_idx] + ".jpg", out_mask)

