import json
import random

def filter_coco_json(input_json, output_json, allowed_classes, max_images=None):
    with open(input_json, "r") as f:
        data = json.load(f)
    
    category_map = {cat["name"]: cat["id"] for cat in data["categories"]}
    allowed_ids = {category_map[name] for name in allowed_classes if name in category_map}
    filtered_categories = [cat for cat in data["categories"] if cat["id"] in allowed_ids]
    new_category_map = {old_id: new_id for new_id, old_id in enumerate(sorted(allowed_ids), start=1)}
    
    for cat in filtered_categories:
        cat["id"] = new_category_map[cat["id"]]
    
    filtered_annotations = [ann for ann in data["annotations"] if ann["category_id"] in allowed_ids]
    for ann in filtered_annotations:
        ann["category_id"] = new_category_map[ann["category_id"]]
    
    used_image_ids = {ann["image_id"] for ann in filtered_annotations}
    filtered_images = [img for img in data["images"] if img["id"] in used_image_ids]
    
    if max_images is not None:
        random.shuffle(filtered_images)
        filtered_images = filtered_images[:max_images]
        allowed_image_ids = {img["id"] for img in filtered_images}
        filtered_annotations = [ann for ann in filtered_annotations if ann["image_id"] in allowed_image_ids]
    
    new_data = {
        "categories": filtered_categories,
        "annotations": filtered_annotations,
        "images": filtered_images
    }
    
    with open(output_json, "w") as f:
        json.dump(new_data, f, indent=4)

# Aerial Maritime Drone
allowed_classes = ["boat", "car"]
input_file = "datasets/odinw13/AerialMaritimeDrone/tiled/train/annotations_without_background.json"  
output_file = "datasets/odinw13/AerialMaritimeDrone/tiled/train/annotations_without_background_overlapped.json"  
filter_coco_json(input_file, output_file, allowed_classes)
input_file = "datasets/odinw13/AerialMaritimeDrone/tiled/test/annotations_without_background.json"  
output_file = "datasets/odinw13/AerialMaritimeDrone/tiled/test/annotations_without_background_overlapped.json"  
filter_coco_json(input_file, output_file, allowed_classes)

allowed_classes = ["car"] 
input_file = "datasets/odinw13/AerialMaritimeDrone/tiled/test/annotations_without_background_overlapped.json"  
output_file = "datasets/odinw13/AerialMaritimeDrone/tiled/test/annotations_without_background_overlapped_filtered.json"
filter_coco_json(input_file, output_file, allowed_classes)

# Hard Hat Workers
allowed_classes = ["person"]
input_file = "datasets/odinw13/HardHatWorkers/raw/train/annotations_without_background.json"
output_file = "datasets/odinw13/HardHatWorkers/raw/train/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)
input_file = "datasets/odinw13/HardHatWorkers/raw/test/annotations_without_background.json"
output_file = "datasets/odinw13/HardHatWorkers/raw/test/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)

allowed_classes = ["person"]
input_file = "datasets/odinw13/HardHatWorkers/raw/test/annotations_without_background.json"  
output_file = "datasets/odinw13/HardHatWorkers/raw/test/annotations_without_background_overlapped_filtered.json"
filter_coco_json(input_file, output_file, allowed_classes)

# Pascal VOC
allowed_classes = ["boat", "car", "dog", "person"]
input_file = "datasets/odinw13/PascalVOC/train/annotations_without_background.json"
output_file = "datasets/odinw13/PascalVOC/train/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)
input_file = "datasets/odinw13/PascalVOC/valid/annotations_without_background.json"
output_file = "datasets/odinw13/PascalVOC/valid/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)

allowed_classes = ["boat"]
input_file = "datasets/odinw13/PascalVOC/valid/annotations_without_background.json"  
output_file = "datasets/odinw13/PascalVOC/valid/annotations_without_background_overlapped_filtered.json"
filter_coco_json(input_file, output_file, allowed_classes)

# Self Driving Cars
allowed_classes = ["car", "truck"]
input_file = "datasets/odinw13/selfdrivingCar/fixedSmall/export/train_annotations_without_background.json"
output_file = "datasets/odinw13/selfdrivingCar/fixedSmall/export/train_annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes, max_images=5000)
input_file = "datasets/odinw13/selfdrivingCar/fixedSmall/export/test_annotations_without_background.json"
output_file = "datasets/odinw13/selfdrivingCar/fixedSmall/export/test_annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)

allowed_classes = ["truck"]
input_file = "datasets/odinw13/selfdrivingCar/fixedSmall/export/test_annotations_without_background.json"  
output_file = "datasets/odinw13/selfdrivingCar/fixedSmall/export/test_annotations_without_background_overlapped_filtered.json"
filter_coco_json(input_file, output_file, allowed_classes)

# Thermal Dogs and People
allowed_classes = ["dog", "person"]
input_file = "datasets/odinw13/thermalDogsAndPeople/train/annotations_without_background.json"
output_file = "datasets/odinw13/thermalDogsAndPeople/train/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)
input_file = "datasets/odinw13/thermalDogsAndPeople/test/annotations_without_background.json"
output_file = "datasets/odinw13/thermalDogsAndPeople/test/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)

allowed_classes = ["dog"]
input_file = "datasets/odinw13/thermalDogsAndPeople/test/annotations_without_background.json"  
output_file = "datasets/odinw13/thermalDogsAndPeople/test/annotations_without_background_overlapped_filtered.json"
filter_coco_json(input_file, output_file, allowed_classes)

# Vehicles Open Images
allowed_classes = ["Car", "Truck"]
input_file = "datasets/odinw13/VehiclesOpenImages/416x416/train/annotations_without_background.json"
output_file = "datasets/odinw13/VehiclesOpenImages/416x416/train/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)
input_file = "datasets/odinw13/VehiclesOpenImages/416x416/test/annotations_without_background.json"
output_file = "datasets/odinw13/VehiclesOpenImages/416x416/test/annotations_without_background_overlapped.json"
filter_coco_json(input_file, output_file, allowed_classes)

allowed_classes = ["Truck"]
input_file = "datasets/odinw13/VehiclesOpenImages/416x416/test/annotations_without_background.json"  
output_file = "datasets/odinw13/VehiclesOpenImages/416x416/test/annotations_without_background_overlapped_filtered.json"
filter_coco_json(input_file, output_file, allowed_classes)