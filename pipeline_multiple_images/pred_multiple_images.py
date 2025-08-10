



def run(arguments):
    try:
        import sys
        from utils.helper import (create_error,
                            report,
                            append_to_coco,
                            export_annotation,
                            save_results)
        from utils.tile_selector import TileSelector
        import os
        import cv2
        from pycocotools.coco import COCO
        from tqdm import tqdm

        

        # Initialize parameters
        input_annotation_path = arguments['input_annotation_path']
        
        images_dir = arguments['images_dir']
        images_name = os.listdir(images_dir)

        tile_size = arguments['tile_size']
        stride = arguments['stride']
        polygon_visibility_threshold=arguments['polygon_visibility_threshold']
        
        output_dir=arguments['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        output_annotation_path = os.path.join(output_dir, 
                                              f"output_{os.path.basename(input_annotation_path)}")

        # Initialize coco annotation
        new_coco_data = {"images":[],
                     "annotations":[],
                     "categories":[]}
        
        coco = COCO(input_annotation_path)

        new_coco_data['categories'] = coco.dataset['categories'] 
        
        for image_name in tqdm(images_name, desc="Processing"):

            image_id = next((image['id'] for image in coco.dataset['images'] if image['file_name']==image_name), None)

            if image_id is None:
                raise ValueError(f"The file {image['file_name']} referenced in the annotation could not be found in the dataset.")

            annotations_ids = coco.getAnnIds(imgIds=image_id)
            image_annotations = coco.loadAnns(annotations_ids)                    
            
            image_path = os.path.join(images_dir,image_name)
            image = cv2.imread(image_path)
            tileselector = TileSelector(image=image,
                                        tile_size=tile_size,
                                        stride=stride,
                                        image_annotations=image_annotations,
                                        polygon_visibility_threshold=polygon_visibility_threshold)
            results = tileselector.run()

            new_coco_data= append_to_coco(coco_data=new_coco_data,
                                    entry=results,
                                    file_name=image_name)
            
            save_results(output_dir=output_dir,
                        entry=results,
                        file_name=image_name)
            
        
        export_annotation(new_coco_data, output_annotation_path)
        
        return report(success=True, result=f'All generated tiles are saved in: {output_dir}')

    except Exception as e :
        exc_type, _, exc_tb = sys.exc_info()
        error = create_error(401, "An error occurred in run function.", str(e), __file__, exc_tb.tb_lineno, exc_type)
        return report(success=False, error=error, summary_code=700)
        
    