import numpy as np
from typing import Tuple, List, Dict
import pyclipper



class TileSelector():
    
    def __init__(self, image: np.ndarray, tile_size: Tuple, stride: Tuple, image_annotations:List, polygon_visibility_threshold:float = 0.8)-> None:
        
        self.image = image
        self.image_annotations = image_annotations
        
        # Set the parameters
        self.tile_size = tile_size
        self.stride = stride
        self.polygon_visibility_threshold = polygon_visibility_threshold


    def __tile_image(self)-> List:
        """
        Splits the input image into overlapping tiles based on tile size and stride.

        This method divides the image into smaller tiles by sliding a window of size 
        `tile_size` over the image with steps defined by `stride`. It ensures full coverage 
        by including tiles at the image edges even if the stride does not fit perfectly.

        Steps:
            1. Determine all possible vertical and horizontal start positions for tiles,
            making sure to include the last tile starting positions to cover the edges.
            2. For each (y_start, x_start) position, extract a tile of size `tile_size` from 
            the image.
            3. Store the tile image data, its unique ID, and the polygon coordinates of the tile 
            in clockwise order: top-left, top-right, bottom-right, bottom-left.
        
        Returns:
            List[dict]: A list of tiles where each tile contains:
                - "id" (int): Unique tile identifier.
                - "data" (np.ndarray): The image data of the tile.
                - "coordinates" (List[int]): List of 8 integers representing the four 
                corner points of the tile polygon in clockwise order:
                [x_start, y_start, x_end, y_start, x_end, y_end, x_start, y_end].
        """
        tiles = []

        image_height, image_width = self.image.shape[:2]
        tile_height, tile_width = self.tile_size
        stride_height, stride_width = self.stride

        max_y = image_height - tile_height
        max_x = image_width - tile_width

        y_starts = list(range(0, max_y + 1, stride_height))
        x_starts = list(range(0, max_x + 1, stride_width))

        if y_starts[-1] != max_y:
            y_starts.append(max_y)
        if x_starts[-1] != max_x:
            x_starts.append(max_x)
        
        tile_id = 0
        
        for y_start in y_starts:
            y_end = y_start + tile_height
            for x_start in x_starts:
                x_end = x_start + tile_width
                
                tiled_image = self.image[y_start:y_end, x_start:x_end]
                tile = {
                    "id": tile_id,
                    "data": tiled_image,
                    "coordinates": [x_start, y_start, x_end, y_start, x_end, y_end, x_start, y_end]}

                tiles.append(tile)
                
                tile_id+=1
        
        return tiles

    def __group_polygons(self, tiles:List)-> List:
        """
        Assigns image annotations to tiles based on polygon overlap and visibility threshold.

        This method processes each tile and determines which annotations belong to it by:
            1. Converting each annotation into a polygon. If `segmentation` is absent, 
            the bounding box is converted into a rectangle polygon.
            2. Computing the polygon intersection between the annotation and the tile.
            3. Calculating the area of both the annotation polygon and the intersection polygon.
            4. Computing the visibility ratio as intersection_area / annotation_area.
            5. Including the annotation in the tile if the visibility ratio is 
            greater than or equal to the configured `polygon_visibility_threshold`.
            6. Adjusting the polygon coordinates of the intersection relative to the tile's top-left corner.

        Inner helper functions:
            - get_area(polygon): Calculates the absolute area of a polygon.
            - get_intersection(polygon1, polygon2): Computes polygon intersection points 
            using pyclipper.
            - adjust_polygon(tile_coordinates, polygon): Translates polygon points to be relative 
            to the tile coordinates.

        Args:
            tiles (List[dict]): List of tile dictionaries, each containing tile image data .

        Returns:
            List[dict]: A list of annotation groups per tile, where each dictionary contains:
                - "tile_id" (int): The tile’s unique identifier.
                - "selected_annotation_ids" (List[int]): Indices of annotations assigned to the tile.
                - "polygons" (List[List[int]]): Polygons of the assigned annotations adjusted 
                to the tile coordinate system.
                - "label_indices" (List[int]): Category IDs corresponding to each assigned annotation.
        """

        def get_area(polygon:List)->int:

            polygon = np.array(polygon, dtype=np.float64).reshape(-1, 2).tolist()
            area = abs(pyclipper.Area(polygon))

            return area

        def get_intersection(polygon1, polygon2)-> List:
            
            polygon1 = np.array(polygon1, dtype=np.float64).reshape(-1, 2).tolist()
            polygon2 = np.array(polygon2, dtype=np.float64).reshape(-1, 2).tolist()

            clipper = pyclipper.Pyclipper()
            clipper.AddPath(polygon1, pyclipper.PT_SUBJECT, True)
            clipper.AddPath(polygon2, pyclipper.PT_CLIP, True)

            intersection = clipper.Execute(pyclipper.CT_INTERSECTION,
                                       pyclipper.PFT_NONZERO,
                                       pyclipper.PFT_NONZERO)
            return intersection 

        def adjust_polygon(tile_coordinates: np.ndarray, polygon: List)-> List:
            
            adjusted_polygon = []
        
            for point in polygon[0]:
                adjusted_polygon.append(point[0] - tile_coordinates[0])
                adjusted_polygon.append(point[1] - tile_coordinates[1])
            
            return adjusted_polygon

        tiles_annotations= []

        for tile in tiles:

            selected_annotation_ids = []
            selected_polygons = []
            selected_label_indices = []

            for selected_id, image_annotation in enumerate(self.image_annotations):
                
                if not image_annotation.get('segmentation'):
                    x_min, y_min, w, h =  image_annotation['bbox']
                    polygon = [x_min, y_min, x_min+w, y_min, x_min+w, y_min+h, x_min, y_min+h]
                else:    
                    polygon = image_annotation['segmentation'][0]
               
                intersection = get_intersection(polygon, tile['coordinates'])
                

                polygon_area = get_area(polygon)   
                intersection_area = get_area(intersection)
                
                polygon_visibility = intersection_area/polygon_area

                if polygon_visibility >= self.polygon_visibility_threshold:
                    
                    adjusted_polygon = adjust_polygon(tile['coordinates'], intersection)
                    
                    selected_polygons.append(adjusted_polygon)
                    selected_annotation_ids.append(selected_id)
                    selected_label_indices.append(image_annotation['category_id'])

            tile_annotations={
                "tile_id" : tile['id'],
                "selected_annotation_ids" : selected_annotation_ids,
                "polygons" : selected_polygons,
                "label_indices" : selected_label_indices}
            
            tiles_annotations.append(tile_annotations)
        
        return tiles_annotations
    
    def __indentify_informative_tiles(self, tiles_annotations: List)-> List:
        """
        Selects a minimal subset of tiles that collectively cover all annotations.

        This method implements a greedy algorithm to identify the smallest set of tiles needed 
        to cover every annotation at least once. The procedure is:

            1. Initialize the set of uncovered annotation IDs to all annotations.
            2. At each iteration, select the tile that covers the largest number of currently 
            uncovered annotations.
            3. Add the selected tile's ID to the list of chosen tiles.
            4. Remove the newly covered annotations from the uncovered set.
            5. Repeat until no uncovered annotations remain or no tile adds new coverage.

        Args:
            tiles_annotations (List[dict]): List of tiles with their assigned annotation IDs, 
                where each dictionary includes:
                    - "tile_id": Tile identifier.
                    - "selected_annotation_ids": List of annotation indices covered by the tile.

        Returns:
            List[int]: List of tile IDs that cover all annotations with minimal redundancy.
        """
        selected_tile_ids = []
        remaining_ids = set(range(len(self.image_annotations)))

        while remaining_ids:
            best_tile = None
            best_new_coverage = set()

            for tile_annotations in tiles_annotations:
                new_covered = set(tile_annotations["selected_annotation_ids"]) & remaining_ids
                if len(new_covered) > len(best_new_coverage):
                    best_new_coverage = new_covered
                    best_tile = tile_annotations

            if best_tile is None:
                break  

            selected_tile_ids.append(best_tile['tile_id'])
            remaining_ids -= best_new_coverage
        
        return selected_tile_ids
    
    def run(self)->Dict:
        """
        Identifies and returns the minimal set of informative image tiles 
        that collectively cover all annotations in the input image.

        This method performs the following steps:
            1. Tiles the input image into smaller overlapping regions (`__tile_image`).
            2. Assigns annotations to each tile based on polygon overlap and visibility 
            threshold (`__group_polygons`).
            3. Selects the minimal set of tiles covering all annotations using a greedy 
            selection algorithm (`__indentify_informative_tiles`).
            4. Filters and returns only the selected tiles and their corresponding annotations.

        Returns:
            Dict: A dictionary containing:
                - "tiles" (List[dict]): List of selected tiles, each with tile ID, image data, 
                and tile polygon coordinates.
                - "tiles_annotations" (List[dict]): Corresponding annotations assigned to 
                each selected tile.
        """
        tiles = self.__tile_image()
        tiles_annotations = self.__group_polygons(tiles)

        filtered_indices = self.__indentify_informative_tiles(tiles_annotations)
        
        informative_tiles = [tile for tile in tiles if tile['id'] in filtered_indices]
        informative_tiles_annotations = [tile_annotations for tile_annotations in tiles_annotations if tile_annotations['tile_id'] in filtered_indices]

        return {"tiles":informative_tiles,
                "tiles_annotations":informative_tiles_annotations}        