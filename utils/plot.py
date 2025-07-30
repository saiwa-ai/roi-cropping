import numpy as np
import cv2
from typing import Dict
import os

def draw_boundarybox(coco_data:Dict, images_dir:str)-> None:
    
    output_dir = os.path.join(images_dir,'annotated_image')
    os.makedirs(output_dir, exist_ok=True)

    n_labels = 0
    
    colors = np.random.rand(len(coco_data['categories']), 3)
    color_map = {category['id']: tuple(color*255) for category, color in zip(coco_data['categories'], colors)}
    
    for image_info in coco_data['images']:

        image_path = os.path.join(images_dir, f"{image_info['file_name']}")
        if not os.path.exists(image_path):
            raise OSError(f"{image_path} does not exist.")
        
        # Load the cropped image
        image = cv2.imread(image_path)

        # Get annotations for the current cropped image
        annotations = [annotation for annotation in coco_data['annotations'] if annotation['image_id'] == image_info['id']]

        if annotations:
            for annotation in annotations:
                if 'segmentation' in annotation:
                    n_labels+=1
                    for segment in annotation['segmentation']:
                        color = color_map.get(annotation['category_id'], (255,255,255))
                        # Create a numpy array from the segmentation points
                        polygon = np.array(segment).reshape((-1, 1, 2)).astype(np.int32)
                        # Draw the polygon on the image
                        cv2.polylines(image, [polygon], isClosed=True, color=color, thickness=6)
                else:
                    raise ValueError("Annotation entry does not contain a 'segmentation' attribute or it is empty.")
        
        output_path = os.path.join(output_dir, f"{image_info['file_name'][:-4]}_with_boundary_boxes.jpg")
        cv2.imwrite(output_path, image)

    print(f"total labels are {n_labels}")
    print(f"annotated images saved in {output_dir}")
    