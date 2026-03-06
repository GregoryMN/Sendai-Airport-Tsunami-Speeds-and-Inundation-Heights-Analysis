---
from ultralytics import YOLO 
 
if __name__ == '__main__': 
    # Load a pretrained detection model 
    model = YOLO("yolo11m.pt")  # Changed from yolo11m-seg.pt 
 
    # Fine-tune on the dataset 
    results = model.train( 
        data="C:/Projects/original_dataset/data.yaml",  # Path to YAML 
        epochs=100,  # Adjust based on time/GPU 
        imgsz=640, 
        batch=16,  # Adjust for your RTX 3070 Ti (8GB VRAM) 
        device=0,  # GPU 
        name="fine_tuned_submerged_detection",  # Updated name 
        amp=True,  # Mixed precision for speed 
        augment=True,  # Enable augmentation 
        workers=4  # Reduced workers to avoid memory issues 
    ) 
 
path: C:/Projects/original_dataset 
train: images/train 
val: images/val 
names: 
  0: level_0 
  1: level_1 
  2: level_2 
  3: level_3 
  4: level_4
---
