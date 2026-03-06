---
from ultralytics import YOLO
import cv2
import numpy as np
import os
import torch  # Import for AMP
import matplotlib.pyplot as plt  # For graph generation
import time
import pandas as pd  # For CSV export

# Start the timer
start_time = time.time()

# Load the fine-tuned detection model and move to GPU
model = YOLO(r"runs\\detect\\fine_tuned_submerged_detection\\weights\\best.pt").to("cuda")
print(f"Model running on: {model.device}")

# Define flood level to height mapping (approximate, based on paper's categories; adjust per literature)
# This mapping is used to convert detected class IDs (0-4) to physical inundation heights in meters.
# Literature basis: [Cite relevant papers on flood level categorization, e.g., from IPCC or similar sources for reproducibility]
FLOOD_HEIGHT_MAP = {
    0: 0.0,   # level_0: No flood
    1: 0.5,   # level_1: Low flood (~0.5 m)
    2: 1.0,   # level_2: Medium flood (~1.0 m)
    3: 1.5,   # level_3: High flood (~1.5 m)
    4: 2.0    # level_4: Full submersion (~2.0 m+)
}

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
output_dir = "runs/detect/exp_height"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "output_video.mp4")

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process frames in batches
batch_size = 50  # Reduced to avoid OOM while still batching
all_heights = []  # For graphs (inundation heights)
all_levels = []  # For graphs (flood levels: 0-4)
all_positions = []  # For graphs (x, y, height)
all_avg_heights = []  # For average height over time
all_std_heights = []  # For error bars in average height plot
with torch.amp.autocast('cuda', enabled=True):  # Updated AMP for efficient GPU use
    for i in range(max(500, 0), frame_count, batch_size):  # Skip first 500 frames
        frames = []
        batch_heights = []  # Per batch for average and std height
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
                    # Create a colored overlay (e.g., blue for water)
                    mask_img = np.zeros_like(frame)
                    cv2.fillPoly(mask_img, [np.array(mask, dtype=np.int32)], (255, 0, 0))  # Blue mask
                    annotated_frame = cv2.addWeighted(annotated_frame, 0.7, mask_img, 0.3, 0)
            
            # Write the annotated frame to output video
            out.write(annotated_frame)
            
            # Extract heights and positions per detection for batch and global
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls)
                    if cls_id in FLOOD_HEIGHT_MAP:
                        height = FLOOD_HEIGHT_MAP[cls_id]
                        batch_heights.append(height)  # Append to batch
                        all_heights.append(height)
                        all_levels.append(cls_id)
                        
                        # Extract position from box center
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x_center = (x1 + x2) / 2
                        y_center = (y1 + y2) / 2
                        all_positions.append((x_center, y_center, height))
        
        # After batch processing, compute avg and std for this batch
        if batch_heights:
            all_avg_heights.append(np.mean(batch_heights))
            all_std_heights.append(np.std(batch_heights))
        else:
            all_avg_heights.append(np.nan)
            all_std_heights.append(np.nan)
        
        batch_heights = []  # Reset for next batch

# Release video writer and cap
out.release()
cap.release()

# Compute time points (batches start after skipped 500 frames)
skip_time = 500 / fps if fps > 0 else 0
time_points = np.arange(len(all_avg_heights)) * (batch_size / fps) + skip_time

# Print for validation
print("Time points (s):", time_points)
print("Average heights (m):", all_avg_heights)
print("Standard deviations (m):", all_std_heights)

# Save to CSV for tables/analysis
df_height = pd.DataFrame({
    'Time': time_points,
    'Avg_Height': all_avg_heights,
    'Std_Height': all_std_heights
})
df_height.to_csv(os.path.join(output_dir, 'height_data.csv'), index=False)
print(f"Height data saved to {os.path.join(output_dir, 'height_data.csv')}")

# Generate visualizations
# 1. Cumulative Histogram of Inundation Heights
plt.figure(figsize=(10, 6))
plt.hist(all_heights, bins=20, cumulative=True, color='skyblue', edgecolor='black', density=True)
plt.axvline(np.mean(all_heights), color='red', linestyle='--', label=f'Mean: {np.mean(all_heights):.2f} m')
plt.axvline(np.mean(all_heights) + np.std(all_heights), color='green', linestyle='--', label=f'Mean + Std: {np.mean(all_heights) + np.std(all_heights):.2f} m')
plt.title('Cumulative Histogram of Inundation Heights', fontsize=14, pad=10)
plt.xlabel('Inundation Height (m)', fontsize=12)
plt.ylabel('Cumulative Probability', fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'height_histogram.png'), dpi=300)
plt.close()

# 2. Line Plot of Average Height Over Time with Error Bars
plt.figure(figsize=(10, 6))
plt.errorbar(time_points, all_avg_heights, yerr=all_std_heights, fmt='o-', color='blue', capsize=5, linewidth=1.5)
plt.title('Average Height Over Time', fontsize=14, pad=10)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Average Height (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'height_over_time.png'), dpi=300)
plt.close()

# 3. Scatter Plot of Height vs. Y-Position
plt.figure(figsize=(10, 6))
x_positions = [p[0] for p in all_positions]
y_positions = [p[1] for p in all_positions]
heights = [p[2] for p in all_positions]
plt.scatter(y_positions, heights, c=heights, cmap='viridis', alpha=0.5)
plt.colorbar(label='Height (m)')
plt.title('Scatter Plot of Height vs. Y-Position', fontsize=14, pad=10)
plt.xlabel('Y-Position (pixels)', fontsize=12)
plt.ylabel('Height (m)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'height_vs_position.png'), dpi=300)
plt.close()

# 4. Pie Chart of Flood Level Distribution
levels = np.arange(5)  # 0 to 4
level_counts = np.bincount(all_levels, minlength=5)
non_zero_mask = level_counts > 0
plt.figure(figsize=(8, 8))
plt.pie(level_counts[non_zero_mask], labels=[f"Level {l}" for l in levels[non_zero_mask]], autopct='%1.1f%%', colors=['#ff9999','#66b3ff','#99ff99','#ffcc99','#ffb3e6'], shadow=True, startangle=90)
plt.title('Flood Level Distribution', fontsize=14, pad=10)
plt.axis('equal')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'flood_level_pie.png'), dpi=300)
plt.close()

# 5. Bar Chart of Detection Counts per Flood Level
plt.figure(figsize=(10, 6))
plt.bar([f"Level {l}" for l in levels], level_counts, color='orange', edgecolor='black')
plt.title('Detection Counts per Flood Level', fontsize=14, pad=10)
plt.xlabel('Flood Level', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'flood_level_bar.png'), dpi=300)
plt.close()

# 6. Heatmap of Vehicle Positions and Inundation Heights
sum_heatmap, xedges, yedges = np.histogram2d([p[0] for p in all_positions], [p[1] for p in all_positions], bins=20, weights=[p[2] for p in all_positions])
count_heatmap, _, _ = np.histogram2d([p[0] for p in all_positions], [p[1] for p in all_positions], bins=20)
avg_heatmap = np.divide(sum_heatmap, count_heatmap, where=count_heatmap != 0, out=np.full_like(sum_heatmap, np.nan))
avg_heatmap = np.ma.masked_where(np.isnan(avg_heatmap), avg_heatmap)  # Mask NaN values (empty bins)
avg_heatmap = avg_heatmap.T  # Transpose for correct orientation

plt.figure()
im = plt.imshow(avg_heatmap, extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], origin='lower', cmap='hot', aspect='auto', vmin=0, vmax=2)
plt.colorbar(im, label='Average Height (m)', shrink=0.8)
plt.title('Heatmap of Vehicle Positions and Inundation Heights', fontsize=14, pad=10)
plt.xlabel('X-Position (pixels)', fontsize=12)
plt.ylabel('Y-Position (pixels)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# Add N/A label for masked (empty) bins
masked_indices = np.where(avg_heatmap.mask)
for idx in range(len(masked_indices[0])):
    y_idx, x_idx = masked_indices[0][idx], masked_indices[1][idx]
    text_color = 'black' if avg_heatmap[y_idx, x_idx] > 1.0 else 'white'
    plt.text(xedges[x_idx] + (xedges[1] - xedges[0]) / 2, yedges[y_idx] + (yedges[1] - yedges[0]) / 2, 
            'N/A', color='black', ha='center', va='center', fontsize=6, alpha=0.7)

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'position_height_heatmap.png'), dpi=300)
plt.close()

# Log detected levels for validation 
with open(os.path.join(output_dir, 'validation_log.txt'), 'w') as log_file:
    unique_levels = np.unique(all_levels)
    log_file.write(f"Detected flood levels: {unique_levels}\n")
    log_file.write(f"Total detections: {len(all_levels)}\n")
    for level in range(5):
        count = all_levels.count(level)
        log_file.write(f"Level {level}: {count} detections\n")
    log_file.write(f"\nTemporal aggregates:\n")
    log_file.write(f"Time points: {time_points.tolist()}\n")
    log_file.write(f"Avg heights: {all_avg_heights}\n")
    log_file.write(f"Std heights: {all_std_heights}\n")

print(f"Processing complete. Time taken: {time.time() - start_time:.2f} seconds")
---
