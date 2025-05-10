import glob
import os
import cv2
import numpy as np

data_name = "images_20250503_115215"

def find_pic_glob(root_dir):
    """
    Find all PNG or JPG files recursively using glob.
    """
    png_files = sorted(glob.glob(os.path.join(root_dir, '**', '*.png'), recursive=True))
    jpg_files = sorted(glob.glob(os.path.join(root_dir, '**', '*.jpg'), recursive=True))
    all_files = png_files + jpg_files
    all_files = sorted(set(all_files))  # Remove duplicates
    return all_files

def load_and_normalize_depth(file_path, mask=None, max_depth_mm=5000.0):
    """
    Load a 16-bit depth image and normalize only values inside the mask to [0, 1].
    
    Parameters:
        file_path (str): Path to the 16-bit PNG depth image.
        mask (np.ndarray or None): Grayscale mask (0-255) where nonzero indicates valid regions.
        max_depth_mm (float): Maximum depth in millimeters to normalize against.

    Returns:
        np.ndarray: Normalized depth map in [0, 1], zero outside the mask.
    """
    depth = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise ValueError(f"Failed to load image: {file_path}")
    if depth.dtype != np.uint16:
        raise ValueError(f"Expected 16-bit PNG: {file_path}")

    depth = depth.astype(np.float32)

    normalized = np.zeros_like(depth, dtype=np.float32)
    if mask is not None:
        depth_masked = depth[mask > 0]
        dmax = depth_masked.max()
        dmin = depth_masked[depth_masked > 0].min() if np.any(depth_masked > 0) else 0
        normalized[mask > 0] = np.clip((depth[mask > 0] - dmin)/ (dmax-dmin), 0.0, 1.0)
    else:
        normalized = np.clip(depth / max_depth_mm, 0.0, 1.0)

    return normalized


def visualize_gray(normalized_depth, window_name, delay_ms=1000):
    """
    Display normalized grayscale depth using OpenCV, auto-advancing after delay_ms.
    """
    gray_8bit = (normalized_depth * 255).astype(np.uint8)
    cv2.imshow(window_name, gray_8bit)
    cv2.waitKey(delay_ms)
    cv2.destroyWindow(window_name)

def main(root_dir, max_depth_mm=1000.0, fps=30):
    mask_files = find_pic_glob(os.path.join(root_dir, 'mask'))
    print(f"Found {len(mask_files)} PNG depth files.")

    window_name = "RGB and Depth Video"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    delay_ms = int(1000 / fps)

    for mask_path in mask_files:
        print(f"Processing: {mask_path}")
        # Paths for RGB and mask
        rgb_path = mask_path.replace('mask', 'rgb')
        depth_path = mask_path.replace('mask', 'depth').replace("jpg", "png")


        # Load RGB and mask
        rgb = cv2.imread(rgb_path)
        if rgb is None:
            print(f"Missing RGB: {rgb_path}")
            continue
        # Load RGB and mask
        rgb = cv2.imread(rgb_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if rgb is None or mask is None:
            print(f"Missing RGB or mask: {rgb_path} or {mask_path}")
            continue
        

        normalized = load_and_normalize_depth(depth_path, mask=mask, max_depth_mm=max_depth_mm)

        # Resize both to the same width (e.g., 640)
        target_width = 640
        target_height = target_width * rgb.shape[0] // rgb.shape[1]
        target_size = (target_width, target_height)

        # Resize RGB and depth
        rgb_resized = cv2.resize(rgb, target_size)
        depth_gray = (normalized * 255).astype(np.uint8)
        depth_bgr = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
        depth_resized = cv2.resize(depth_bgr, target_size)

        # Optional: Add labels
        cv2.putText(rgb_resized, 'RGB', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(depth_resized, 'Depth', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        # Stack vertically
        combined = np.vstack((rgb_resized, depth_resized))

        cv2.imshow(window_name, combined)

        if cv2.waitKey(delay_ms) & 0xFF == 27:  # ESC to quit early
            break

    cv2.destroyAllWindows()



# Example usage
if __name__ == "__main__":
    main(f"demo_data/{data_name}/", max_depth_mm=1000.0, fps=30)
