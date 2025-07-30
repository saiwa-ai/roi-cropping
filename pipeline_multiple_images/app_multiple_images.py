from pipeline_multiple_images import pred_multiple_images
import sys
from utils.helper import create_error, report
import os


def run(arguments):
    
    if not os.path.exists(str(arguments['input_annotation_path'])): 
        error = create_error(101, "input_annotation_path does not exist.", arguments['input_annotation_path'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)
    
    if not os.path.isdir(arguments['images_dir']):
        error = create_error(102, "images_dir  does not exist.", arguments['images_dir'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)
    
    if type(arguments['tile_size']) != list :
        error = create_error(104, "tile_size should be a list.", arguments['tile_size'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)

    if type(arguments['stride']) != list :
        error = create_error(104, "stride should be a list.", arguments['stride'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)
            
    try:
        arguments['polygon_visibility_threshold'] = float(arguments['polygon_visibility_threshold']) or 0.8
    except:
        error = create_error(104, "polygon_visibility_threshold should be a float number.", arguments['polygon_visibility_threshold'], __file__, sys._getframe().f_lineno)
        return report(success=False, error=error, summary_code=700)
     
    return pred_multiple_images.run(arguments)

def handler(event, context):
    try:
        return run(event)
    except Exception as e:
        exe_type, _, exc_tb = sys.exc_info()
        error = create_error(401, "An error occurred in handler function.", str(e), __file__, exc_tb.tb_lineno, exe_type)
        return report(success=False, error=error , summary_code=700)