from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch  # Import for AMP
from skimage.metrics import structural_similarity as ssim  # For symmetry score
import tensorflow as tf  # For TFLite inference
import matplotlib.pyplot as plt  # For graph generation
import time
from scipy.stats import linregress
import pandas as pd  # For CSV export

# Start the timer
start_time = time.time()

# Load the local YOLO11m segmentation model and move to GPU
model = YOLO("yolo11m-seg.pt").to("cuda")
print(f"Model running on: {model.device}")

# Load the TFLite orientation model
tflite_model_path = "vehicle_orientation.tflite"
interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Define class mapping from TFLite model (15 classes from repo)
ORIENTATION_CLASSES = [
    "car_front", "car_side", "car_back",
    "bus_front", "bus_side", "bus_back",
    "truck_front", "truck_side", "truck_back",
    "motorcycle_front", "motorcycle_side", "motorcycle_back",
    "bicycle_front", "bicycle_side", "bicycle_back"
]

# Define average vehicle dimensions in Japan (meters)
CAR_LENGTH = 4.7  # Average length for cars (side view)
CAR_WIDTH = 1.7   # Average width for cars (front/back view)
BUS_LENGTH = 12.0  # Average length for buses (side view, midpoint of 6m)
BUS_WIDTH = 2.5   # Average width for buses (front/back view)

SUBMERGED_EXTENSION_FACTOR = 0.3  # Extend mask downward by 30% of box height to fill submerged area

class KalmanFilter:
    def __init__(self):
        # State vector [x, y, vx, vy]
        self.state = np.zeros((4, 1), dtype=np.float32)
        # Transition matrix (constant velocity model)
        self.transition_matrix = np.eye(4, dtype=np.float32)
        self.transition_matrix[0, 2] = 1.0  # dt=1
        self.transition_matrix[1, 3] = 1.0
        # Measurement matrix (observe x, y)
        self.measurement_matrix = np.eye(2, 4, dtype=np.float32)
        # Process noise covariance
        self.process_noise_cov = np.eye(4, dtype=np.float32) * 0.01
        # Measurement noise covariance
        self.measurement_noise_cov = np.eye(2, dtype=np.float32) * 0.1
        # Error covariance
        self.error_cov_post = np.eye(4, dtype=np.float32) * 1.0

    def predict(self):
        self.state = np.dot(self.transition_matrix, self.state)
        self.error_cov_post = np.dot(self.transition_matrix, np.dot(self.error_cov_post, self.transition_matrix.T)) + self.process_noise_cov
        return self.state[0:2].flatten()  # Predicted position

    def update(self, measurement):
        measurement = np.array(measurement, dtype=np.float32).reshape(2, 1)
        y = measurement - np.dot(self.measurement_matrix, self.state)
        S = np.dot(self.measurement_matrix, np.dot(self.error_cov_post, self.measurement_matrix.T)) + self.measurement_noise_cov
        K = np.dot(np.dot(self.error_cov_post, self.measurement_matrix.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, y)
        I_KH = np.eye(4) - np.dot(K, self.measurement_matrix)
        self.error_cov_post = np.dot(I_KH, self.error_cov_post)
        return self.state[2:4].flatten()  # Velocity

# Open the input video
cap = cv2.VideoCapture("2011 Japan Tsunami - Sendai Airport Terminal. (Full Footage)_1080p.mp4")
if not cap.isOpened():
    raise ValueError("Cannot open video file")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video properties - Width: {frame_width}, Height: {frame_height}, FPS: {fps}, Total frames: {frame_count}")

# Define output video path
output_dir = "runs/detect/exp_speed"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_video.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process frames in batches
batch_size = 50  # Reduced to avoid OOM while still batching
kalman_filters = {}  # Kalman filter per track ID
track_durations = {}  # Track duration per ID for filtering
mpp_history = {}  # MPP history per track for averaging
prev_frame_gray = None  # For optical flow
all_speeds = []  # For graphs (speeds)
all_orientations = []  # For graphs (orientations: front/back/side/unknown)
all_types = []  # For graphs (vehicle types: car/bus/unknown)
all_positions = []  # For graphs (x, y, speed)
all_avg_speeds = []  # For average speed over time
all_std_speeds = []  # For error bars in average speed plot
with torch.amp.autocast('cuda', enabled=True):  # Updated AMP for efficient GPU use
    for i in range(max(500, 0), frame_count, batch_size):  # Skip first 500 frames
        frames = []
        batch_speeds = []  # Per batch for average and std speed
        for _ in range(min(batch_size, frame_count - i)):
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        if not frames:
            break
        
        # Predict on the batch with lower confidence and balanced resolution
        results = model.predict(source=frames, conf=0.4, iou=0.5, tracker="botsort.yaml", save=False, imgsz=640)  # Balanced resolution
        for result, frame in zip(results, frames):
            # Get annotated frame (with boxes and labels, but without masks overlaid yet)
            annotated_frame = result.plot(labels=True, masks=False)  # Disable default mask plotting to avoid dimming

            # Apply 30% opaque colored mask only to segmented areas
            if result.masks is not None:
                masks = result.masks.xy
                for mask in masks:
                    # Create a colored overlay (e.g., blue with alpha 0.3)
                    overlay = np.zeros_like(frame)
                    cv2.fillPoly(overlay, [np.int32(mask)], (255, 0, 0))  # Blue (BGR format)
                    # Blend only the masked area
                    mask_binary = np.zeros_like(frame[:,:,0])
                    cv2.fillPoly(mask_binary, [np.int32(mask)], 255)
                    mask_binary = mask_binary > 0
                    annotated_frame[mask_binary] = cv2.addWeighted(annotated_frame[mask_binary], 0.7, overlay[mask_binary], 0.3, 0)

            # Calculate speed and intelligent orientation for detected vehicles
            boxes = result.boxes.xyxy
            current_positions = {}
            class_names = result.names  # Get class names from model
            ids = result.boxes.id if result.boxes.id is not None else range(len(boxes))  # Use track IDs if available
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Define curr_gray here
            for j, box in enumerate(boxes):
                track_id = int(ids[j]) if ids is not None else j
                x1, y1, x2, y2 = map(int, box[:4])
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                class_id = int(result.boxes.cls[j]) if result.boxes.cls is not None else 0
                vehicle_type = class_names[class_id] if class_id in class_names and class_names[class_id] in ["car", "bus"] else "unknown"
                confidence = float(result.boxes.conf[j]) if result.boxes.conf is not None else 1.0
                # Intelligent orientation detection using local TFLite model
                view = "unknown"
                if confidence > 0.2:  # Classify low-confidence detections
                    crop = frame[y1:y2, x1:x2]
                    h_crop = y2 - y1
                    extension_height = int(h_crop * SUBMERGED_EXTENSION_FACTOR)
                    extended_crop = np.ones((h_crop + extension_height, x2 - x1, 3), dtype=np.uint8) * 255  # White extension
                    extended_crop[0:h_crop, :] = crop
                    # Resize to input shape (640x640 for this model)
                    resized_crop = cv2.resize(extended_crop, (640, 640))
                    resized_crop = resized_crop.astype(np.float32) / 255.0  # Normalize
                    resized_crop = np.expand_dims(resized_crop, axis=0)  # Add batch dimension
                    interpreter.set_tensor(input_details[0]['index'], resized_crop)
                    interpreter.invoke()
                    output_data = interpreter.get_tensor(output_details[0]['index'])[0]  # Shape [25200, 20]
                    # Parse YOLO output: find max conf detection
                    confs = output_data[:, 4]
                    max_conf_idx = np.argmax(confs)
                    max_conf = confs[max_conf_idx]
                    if max_conf > 0.2:  # Low threshold
                        class_probs = output_data[max_conf_idx, 5:]  # Classes probabilities
                        class_id = np.argmax(class_probs)
                        ori_conf = class_probs[class_id]
                        api_view = ORIENTATION_CLASSES[class_id]
                        if "front" in api_view or "back" in api_view:
                            view = "front/back"
                        elif "side" in api_view:
                            view = "side"
                # Fallback to mask if local model confidence low
                if view == "unknown" and result.masks is not None and j < len(result.masks.xy):
                    mask = result.masks.xy[j]
                    mask_array = np.array(mask, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(mask_array)
                    mask_img = np.zeros((h, w), dtype=np.uint8)
                    cv2.fillPoly(mask_img, [mask_array - [x, y]], 255)
                    # Vertical symmetry check (for front/back)
                    if h % 2 != 0:
                        mask_img = mask_img[:-1, :]
                    top = mask_img[:h//2, :]
                    bottom = np.flipud(mask_img[h//2:, :])
                    symmetry_score = ssim(top, bottom, multichannel=False) if top.size > 0 else 0.0
                    mask_aspect_ratio = w / h if h > 0 else 1.0
                    # Typical mask aspect ratios: side view ~2.5-3.0, front/back ~1.0-1.5
                    expected_ratio_side = (CAR_LENGTH / CAR_WIDTH) if vehicle_type == "car" else (BUS_LENGTH / BUS_WIDTH)
                    side_diff = abs(mask_aspect_ratio - expected_ratio_side)
                    if side_diff < 0.8 and mask_aspect_ratio > 2.0 and symmetry_score < 0.6:
                        view = "side"
                    elif 0.8 < mask_aspect_ratio < 1.5 and symmetry_score > 0.8:
                        view = "front/back"
                # Set reference dimensions based on inferred view (group left/right as side)
                if vehicle_type in ["car", "bus", "unknown"]:
                    reference_length = CAR_LENGTH if vehicle_type == "car" else BUS_LENGTH if vehicle_type == "bus" else CAR_LENGTH
                    reference_width = CAR_WIDTH if vehicle_type == "car" else BUS_WIDTH if vehicle_type == "bus" else CAR_WIDTH
                    reference_dim = reference_length if view == "side" else reference_width
                else:
                    reference_dim = CAR_WIDTH  # Default
                # Dynamic mpp from vehicle size
                pixel_size = (x2 - x1) if view == "side" else (y2 - y1)
                mpp = reference_dim / pixel_size if pixel_size > 0 else 0.1
                # Average mpp over last 5 frames for stability
                if track_id not in mpp_history:
                    mpp_history[track_id] = []
                mpp_history[track_id].append(mpp)
                if len(mpp_history[track_id]) > 5:
                    mpp_history[track_id].pop(0)
                avg_mpp = np.mean(mpp_history[track_id])
                # Perspective adjustment based on y-position (higher y = farther)
                depth_factor = 1.0 + (center_y / frame_height) * 0.2  # Adjust 0.2 for your footage's perspective
                avg_mpp *= depth_factor
                current_positions[track_id] = (center_x, center_y)
                # Calculate displacement and speed with Kalman filter and optical flow fusion
                if track_id not in kalman_filters:
                    kalman_filters[track_id] = KalmanFilter()
                    track_durations[track_id] = 0
                kf = kalman_filters[track_id]
                predicted_pos = kf.predict()
                kf.update([center_x, center_y])
                velocity = kf.update([center_x, center_y])  # Get velocity from update
                pixel_speed = np.sqrt(velocity[0]**2 + velocity[1]**2)
                speed_mps = pixel_speed * avg_mpp * 2.5  # Apply correction factor
                # Optical flow fusion if prev frame available
                if prev_frame_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray[y1:y2, x1:x2], curr_gray[y1:y2, x1:x2], None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    avg_flow = np.mean(flow, axis=(0, 1))
                    flow_speed = np.sqrt(avg_flow[0]**2 + avg_flow[1]**2)
                    speed_mps = 0.7 * speed_mps + 0.3 * (flow_speed * avg_mpp)  # Fuse with Kalman speed
                track_durations[track_id] += 1
                # Collect data for graphs if track is stable
                if track_durations[track_id] >= 5 and 1.0 < speed_mps < 17:
                    batch_speeds.append(speed_mps)  # Per batch collection
                    all_speeds.append(speed_mps)
                    all_orientations.append(view)
                    all_types.append(vehicle_type)
                    all_positions.append((center_x, center_y, speed_mps))
                # Display details on top right of box if track is stable
                if track_durations[track_id] >= 5 and 0 < speed_mps < 50:  # Clip unrealistic speeds
                    label = f"{vehicle_type} ({view})\nSpeed: {speed_mps:.2f} m/s"
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0][0]
                    cv2.putText(annotated_frame, label, (x2 - text_size - 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            # Update prev frame for optical flow
            prev_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Write the frame to video
            out.write(annotated_frame)
            print(f"Batch {i//batch_size + 1}, Detected boxes: {boxes.shape[0]}")
            if result.masks is not None:
                print(f"Segmentation masks: {result.masks.data.shape}")

# Release resources
cap.release()
out.release()
print(f"Video saved to {output_path}")

# Stop the timer and calculate elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
hours, rem = divmod(elapsed_time, 3600)
minutes, seconds = divmod(rem, 60)
print(f"Video analysis completed in {int(hours):02d}:{int(minutes):02d}:{int(seconds):02d} (hh:mm:ss)")

# Clean up individual frame saves if any
for file in os.listdir(output_dir):
    if file.endswith((".jpg", ".png")):
        os.remove(os.path.join(output_dir, file))

# Generate graphs if data exists
if all_speeds:
    # Compute batch averages and std deviations
    for j in range(0, len(all_speeds), batch_size):
        batch = all_speeds[j:j + batch_size]
        if batch:
            all_avg_speeds.append(np.mean(batch))
            all_std_speeds.append(np.std(batch))
        else:
            all_avg_speeds.append(np.nan)
            all_std_speeds.append(np.nan)

    # Compute time points (batches start after skipped 500 frames)
    skip_time = 500 / fps if fps > 0 else 0
    time_points = np.arange(len(all_avg_speeds)) * (batch_size / fps) + skip_time

    # Print for validation
    print("Time points (s):", time_points)
    print("Average speeds (m/s):", all_avg_speeds)
    print("Standard deviations (m/s):", all_std_speeds)

    # Save to CSV for tables/analysis
    df_speed = pd.DataFrame({
        'Time': time_points,
        'Avg_Speed': all_avg_speeds,
        'Std_Speed': all_std_speeds
    })
    df_speed.to_csv(os.path.join(output_dir, 'speed_data.csv'), index=False)
    print(f"Speed data saved to {os.path.join(output_dir, 'speed_data.csv')}")

    # 1. Histogram of Vehicle Speeds
    plt.figure(figsize=(10, 6))
    plt.hist(all_speeds, bins=20, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(all_speeds), color='r', linestyle='dashed', linewidth=1, label=f'Mean: {np.mean(all_speeds):.2f} m/s')
    plt.axvline(np.mean(all_speeds) + np.std(all_speeds), color='g', linestyle='dashed', linewidth=1, label=f'Mean + Std Dev: {np.mean(all_speeds) + np.std(all_speeds):.2f} m/s')
    plt.title('Histogram of Vehicle Speeds', fontsize=14, pad=10)
    plt.xlabel('Speed (m/s)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_histogram.png'), dpi=300)
    plt.close()

    # 2. Line Plot of Average Speed Over Time with Error Bars
    plt.figure(figsize=(12, 6))
    plt.errorbar(time_points, all_avg_speeds, yerr=all_std_speeds, fmt='o-', color='blue', capsize=5, linewidth=1.5)
    plt.title('Average Speed Over Time', fontsize=14, pad=10)
    plt.xlabel('Time (s)', fontsize=12)
    plt.ylabel('Avg Speed (m/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_over_time.png'), dpi=300)
    plt.close()

    # 3. Scatter Plot of Speed vs. Position
    y_positions = [p[1] for p in all_positions]
    speeds = [p[2] for p in all_positions]
    plt.figure(figsize=(10, 6))
    plt.scatter(y_positions, speeds, c=speeds, cmap='viridis', alpha=0.5)
    plt.title('Speed vs. Y-Position (Distance Proxy)', fontsize=14, pad=10)
    plt.xlabel('Y-Position (pixels)', fontsize=12)
    plt.ylabel('Speed (m/s)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.colorbar(label='Speed (m/s)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_vs_position.png'), dpi=300)
    plt.close()

    # 4. Pie Chart of Vehicle Orientation Distribution
    orientation_counts = np.unique(all_orientations, return_counts=True)
    plt.figure(figsize=(8, 8))
    plt.pie(orientation_counts[1], labels=orientation_counts[0], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99'], shadow=True, startangle=180)
    plt.title('Vehicle Orientation Distribution', fontsize=14, pad=10)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'orientation_pie.png'), dpi=300)
    plt.close()

    # 5. Bar Chart of Detection Counts per Vehicle Type
    type_counts = np.unique(all_types, return_counts=True)
    plt.figure(figsize=(10, 6))
    plt.bar(type_counts[0], type_counts[1], color='orange', edgecolor='black')
    plt.title('Detection Counts per Vehicle Type', fontsize=14, pad=10)
    plt.xlabel('Vehicle Type', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'vehicle_type_bar.png'), dpi=300)
    plt.close()

    # 6. Heatmap of Vehicle Positions and Speeds
    # Summed speeds
    heatmap_sum, xedges, yedges = np.histogram2d([p[0] for p in all_positions], [p[1] for p in all_positions], bins=20, weights=[p[2] for p in all_positions])
    # Detection counts per bin
    heatmap_count, _, _ = np.histogram2d([p[0] for p in all_positions], [p[1] for p in all_positions], bins=20)
    # Average speeds (divide sum by count, handle zero-division with masking)
    heatmap_avg = np.divide(heatmap_sum, heatmap_count, where=heatmap_count != 0, out=np.full_like(heatmap_sum, np.nan))
    heatmap_avg = np.ma.masked_where(np.isnan(heatmap_avg), heatmap_avg)  # Mask NaN values (empty bins)
    heatmap_avg = heatmap_avg.T  # Transpose for correct orientation

    plt.figure()
    im = plt.imshow(heatmap_avg, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='hot', aspect='auto', vmin=0, vmax=17)
    plt.colorbar(im, label='Average Speed (m/s)', shrink=0.8)
    plt.title('Heatmap of Vehicle Positions and Speeds', fontsize=14, pad=10)
    plt.xlabel('X-Position (pixels)', fontsize=12)
    plt.ylabel('Y-Position (pixels)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add text label for masked (empty) bins
    masked_indices = np.where(heatmap_avg.mask)
    for idx in range(len(masked_indices[0])):
        y_idx, x_idx = masked_indices[0][idx], masked_indices[1][idx]
        plt.text(xedges[x_idx] + (xedges[1] - xedges[0]) / 2, yedges[y_idx] + (yedges[1] - yedges[0]) / 2, 
                'N/A', color='black', ha='center', va='center', fontsize=6, alpha=0.7)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'position_speed_heatmap.png'), dpi=300)
    plt.close()

# Calculate mean speeds for cars and buses
car_speeds = [speed for speed, typ in zip(all_speeds, all_types) if typ == "car"]
bus_speeds = [speed for speed, typ in zip(all_speeds, all_types) if typ == "bus"]
mean_car = np.mean(car_speeds) if car_speeds else 0.0
mean_bus = np.mean(bus_speeds) if bus_speeds else 0.0

# Validation log for detection and speed
with open(os.path.join(output_dir, 'speed_validation_log.txt'), 'w') as log_file:
    log_file.write(f"Total detections: {len(all_speeds)}\n")
    log_file.write(f"Mean speed: {np.mean(all_speeds):.2f} m/s, Median: {np.median(all_speeds):.2f} m/s, Range: {min(all_speeds):.2f}-{max(all_speeds):.2f} m/s (capped at 17 m/s)\n")
    log_file.write(f"Mean speed of cars: {mean_car:.2f} m/s\n")
    log_file.write(f"Mean speed of buses: {mean_bus:.2f} m/s\n")
    log_file.write(f"Expected range (tsunami flow): 3-17 m/s\n")
    log_file.write(f"\nTemporal aggregates:\n")
    log_file.write(f"Time points: {time_points.tolist()}\n")
    log_file.write(f"Avg speeds: {all_avg_speeds}\n")
    log_file.write(f"Std speeds: {all_std_speeds}\n")

print(f"Processing complete. Time taken: {time.time() - start_time:.2f} seconds")