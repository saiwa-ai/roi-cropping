from typing import List, Dict
import os
import cv2
import json

def report(success=True, result=None, error=None, summary_code=200):
    summary={
        "success" : success,
        "result" : result,
        "error" : error
    }

    return summary

def create_error(code, message, details, filename, lineno, type=None):
    
    error = {
        "code" : code,
        "message" : message,
        "devel" : {
            "details" : details,
            "filename" : filename,
            "line" : lineno,
            "type" : str(type)
        }
    }

    return error

def append_to_coco(coco_data: Dict, entry:Dict, file_name:str)-> Dict:

    def convert_polygon_to_bbox(polygon:List)->List:
        
        xs = polygon[0::2]
        ys = polygon[1::2]

        x_min, y_min = min(xs), min(ys)
        x_max, y_max = max(xs), max(ys)

        return [x_min, y_min, x_max-x_min, y_max-y_min]

    image_id = len(coco_data['images'])
    annotation_id = len(coco_data['annotations'])

    for tile in entry['tiles']:
        
        tile_height, tile_width = tile['data'].shape[:2]

        image_info = {"id":image_id,
                      "file_name":f"{file_name[:-4]}_{tile['id']}.jpg",
                      "height":tile_height,
                      "width":tile_width}
        
        coco_data['images'].append(image_info)

        for tile_annotations in entry['tiles_annotations']:
            if tile_annotations['tile_id']==tile['id']:
                for polygon, label_index in zip(tile_annotations['polygons'], tile_annotations['label_indices']):   
                    
                    bbox = convert_polygon_to_bbox(polygon)
                    
                    annotation_info = {"id": annotation_id,
                                       "image_id": image_id,
                                       "category_id": label_index,
                                       "bbox":bbox,
                                       "area":bbox[2]*bbox[3],
                                       "segmentation":[polygon],
                                       "iscrowd":0}
                    
                    coco_data['annotations'].append(annotation_info)
                    annotation_id+=1
        image_id+=1

    return coco_data

    
def save_result(output_dir:str, entry:Dict, file_name:str):
    
    tiles = entry['tiles']
    for tile in tiles:
        output_path=os.path.join(output_dir, f"{file_name[:-4]}_{tile['id']}.jpg")
        cv2.imwrite(output_path, tile['data'])


def export_annotation(data:Dict, output_annotation_path:str):

    with open(output_annotation_path, 'w') as f:
        json.dump(data,f, indent=4)