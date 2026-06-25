import os
import xml.etree.ElementTree as ET
import cv2
import random
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

CLASSES = ["D00", "D20", "D40"]

def get_iou(boxA, boxB):
    # box format: [xmin, ymin, xmax, ymax]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if boxAArea + boxBArea - interArea == 0:
        return 0
        
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def check_overlap_with_any(candidate_box, ground_truth_boxes):
    for gt_box in ground_truth_boxes:
        if get_iou(candidate_box, gt_box) > 0.0:
            return True
    return False

def parse_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except Exception as e:
        print(f"Error parsing XML {xml_path}: {e}")
        return None, []
        
    filename = root.find("filename").text
    
    # Get image size
    size_node = root.find("size")
    width = int(size_node.find("width").text)
    height = int(size_node.find("height").text)
    
    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text
        if name in CLASSES:
            bndbox = obj.find("bndbox")
            xmin = int(float(bndbox.find("xmin").text))
            ymin = int(float(bndbox.find("ymin").text))
            xmax = int(float(bndbox.find("xmax").text))
            ymax = int(float(bndbox.find("ymax").text))
            
            # Make sure coordinates are within image boundaries
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(width, xmax)
            ymax = min(height, ymax)
            
            if xmax > xmin and ymax > ymin:
                objects.append({
                    "class": name,
                    "box": [xmin, ymin, xmax, ymax]
                })
                
    return filename, objects

def process_dataset():
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    
    # Locate images and annotations
    # RDD2022 India zip usually contains India/images/ and India/annotations/xmls/
    india_dir = None
    for p in raw_dir.glob("**/annotations/xmls"):
        india_dir = p.parent.parent
        break
        
    if not india_dir:
        print("Could not find India annotations directory. Please check extraction path.")
        return
        
    images_dir = india_dir / "images"
    annotations_dir = india_dir / "annotations" / "xmls"
    
    print(f"Found dataset at: {india_dir}")
    xml_files = list(annotations_dir.glob("*.xml"))
    print(f"Total xml annotation files: {len(xml_files)}")
    
    # Split xml files at image level to avoid data leakage
    train_xml, test_xml = train_test_split(xml_files, test_size=0.3, random_state=42)
    val_xml, test_xml = train_test_split(test_xml, test_size=0.5, random_state=42)
    
    splits = {
        "train": train_xml,
        "val": val_xml,
        "test": test_xml
    }
    
    # Initialize output class directories
    for split in splits:
        for cls in CLASSES + ["Normal"]:
            (processed_dir / split / cls).mkdir(parents=True, exist_ok=True)
            
    # Process each split
    for split_name, files in splits.items():
        print(f"Processing {split_name} split ({len(files)} images)...")
        
        for xml_path in tqdm(files):
            img_filename, objects = parse_xml(xml_path)
            if img_filename is None:
                continue
                
            img_path = images_dir / img_filename
            # Fallback in case of extensions mismatches
            if not img_path.exists():
                img_path = images_dir / (xml_path.stem + ".jpg")
                
            if not img_path.exists():
                continue
                
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w, _ = img.shape
            gt_boxes = [obj["box"] for obj in objects]
            
            # 1. Crop actual damage patches
            for idx, obj in enumerate(objects):
                cls_name = obj["class"]
                box = obj["box"]
                xmin, ymin, xmax, ymax = box
                
                # Crop and save patch
                patch = img[ymin:ymax, xmin:xmax]
                if patch.size > 0:
                    patch_name = f"{xml_path.stem}_damage_{idx}_{cls_name}.jpg"
                    out_path = processed_dir / split_name / cls_name / patch_name
                    cv2.imwrite(str(out_path), patch)
            
            # 2. Sample random Normal patches
            # Try to crop a few non-overlapping background patches of average damage size (e.g. 128x128)
            patch_size = 128
            sampled_normal = 0
            # Limit the number of normal patches per image to keep classes balanced
            max_normal_per_image = max(1, len(objects))
            
            attempts = 0
            while sampled_normal < max_normal_per_image and attempts < 20:
                attempts += 1
                if w - patch_size <= 0 or h - patch_size <= 0:
                    break
                nx = random.randint(0, w - patch_size)
                ny = random.randint(0, h - patch_size)
                candidate_box = [nx, ny, nx + patch_size, ny + patch_size]
                
                if not check_overlap_with_any(candidate_box, gt_boxes):
                    patch = img[ny:ny+patch_size, nx:nx+patch_size]
                    if patch.size > 0:
                        patch_name = f"{xml_path.stem}_normal_{sampled_normal}.jpg"
                        out_path = processed_dir / split_name / "Normal" / patch_name
                        cv2.imwrite(str(out_path), patch)
                        sampled_normal += 1
                        
    print("Patch preprocessing completed!")
    # Show stats
    for split in splits:
        print(f"\nStats for {split}:")
        for cls in CLASSES + ["Normal"]:
            path = processed_dir / split / cls
            count = len(list(path.glob("*.jpg")))
            print(f"  - {cls}: {count} patches")

if __name__ == "__main__":
    process_dataset()
