"""
Form Field Extraction processor for identifying and extracting structured data from forms.
Identifies form fields, labels, and their corresponding values.
"""

import os
import cv2
import numpy as np
import re
import json
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path

class FormFieldExtractor:
    """
    Processor for extracting form fields and their values from document images.
    Uses a combination of computer vision and OCR to identify form structure.
    """
    
    def __init__(self, ocr_processor=None):
        """
        Initialize the form field extractor.
        
        Args:
            ocr_processor: OCR processor for text extraction
        """
        self.ocr_processor = ocr_processor
    
    def extract_fields(self, image: np.ndarray, text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract form fields and their values from an image.
        
        Args:
            image: Input image
            text_regions: OCR results with text and bounding boxes
            
        Returns:
            Dictionary with extracted form fields and values
        """
        # Find form elements (checkboxes, text fields, etc.)
        form_elements = self._detect_form_elements(image)
        
        # Associate OCR text with form elements
        fields = self._associate_text_with_elements(form_elements, text_regions)
        
        # Post-process and structure the extracted fields
        result = self._structure_fields(fields)
        
        return result
    
    def _detect_form_elements(self, image: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """
        Detect form elements in the image using computer vision techniques.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with different types of form elements
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Initialize result containers
        elements = {
            "checkboxes": [],
            "text_fields": [],
            "radio_buttons": [],
            "lines": [],
            "tables": []
        }
        
        # Detect horizontal and vertical lines (form structure)
        horizontal_lines, vertical_lines = self._detect_lines(binary)
        
        # Store line information
        if horizontal_lines is not None:
            for line in horizontal_lines:
                x1, y1, x2, y2 = line[0]
                elements["lines"].append({
                    "type": "horizontal",
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                })
                
        if vertical_lines is not None:
            for line in vertical_lines:
                x1, y1, x2, y2 = line[0]
                elements["lines"].append({
                    "type": "vertical",
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2)
                })
        
        # Detect checkboxes and radio buttons
        elements["checkboxes"] = self._detect_checkboxes(binary)
        elements["radio_buttons"] = self._detect_radio_buttons(binary)
        
        # Detect text fields using line intersections
        elements["text_fields"] = self._detect_text_fields(
            horizontal_lines, vertical_lines, image.shape[1], image.shape[0]
        )
        
        # Detect tables based on grid patterns
        elements["tables"] = self._detect_tables(
            horizontal_lines, vertical_lines, image.shape[1], image.shape[0]
        )
        
        return elements
    
    def _detect_lines(self, binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect horizontal and vertical lines in a form.
        
        Args:
            binary: Binary image
            
        Returns:
            Tuple of (horizontal_lines, vertical_lines)
        """
        # Set minimum line length based on image dimensions
        height, width = binary.shape
        min_line_length = min(height, width) // 10
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_detect = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.HoughLinesP(
            horizontal_detect, 1, np.pi/180, 
            threshold=100, minLineLength=min_line_length, maxLineGap=10
        )
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_detect = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_lines = cv2.HoughLinesP(
            vertical_detect, 1, np.pi/180, 
            threshold=100, minLineLength=min_line_length, maxLineGap=10
        )
        
        return horizontal_lines, vertical_lines
    
    def _detect_checkboxes(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect checkboxes in a form.
        
        Args:
            binary: Binary image
            
        Returns:
            List of detected checkboxes with coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        checkboxes = []
        
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if the contour has roughly the shape of a square
            aspect_ratio = float(w) / h
            
            # Checkboxes are typically square with small size
            if 0.8 <= aspect_ratio <= 1.2 and 10 <= w <= 30 and 10 <= h <= 30:
                # Check if filled
                roi = binary[y:y+h, x:x+w]
                total_pixels = roi.shape[0] * roi.shape[1]
                filled_pixels = cv2.countNonZero(roi)
                filled_ratio = filled_pixels / total_pixels
                
                is_checked = filled_ratio > 0.5  # More than half filled
                
                checkboxes.append({
                    "x": x,
                    "y": y,
                    "width": w,
                    "height": h,
                    "is_checked": is_checked,
                    "filled_ratio": filled_ratio
                })
        
        return checkboxes
    
    def _detect_radio_buttons(self, binary: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect radio buttons in a form.
        
        Args:
            binary: Binary image
            
        Returns:
            List of detected radio buttons with coordinates
        """
        # Find contours
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        radio_buttons = []
        
        for contour in contours:
            # Calculate contour properties
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            
            # Circle has minimum perimeter for a given area
            # Circularity = 4π(area/perimeter²)
            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # Circles have circularity close to 1
                if 0.8 <= circularity <= 1.2 and 10 <= area <= 500:
                    # Fit a circle
                    center, radius = cv2.minEnclosingCircle(contour)
                    x, y = int(center[0]), int(center[1])
                    radius = int(radius)
                    
                    # Check if filled
                    mask = np.zeros(binary.shape, np.uint8)
                    cv2.circle(mask, (x, y), max(radius-3, 1), 255, -1)
                    masked_roi = cv2.bitwise_and(binary, binary, mask=mask)
                    
                    total_pixels = cv2.countNonZero(mask)
                    filled_pixels = cv2.countNonZero(masked_roi)
                    filled_ratio = filled_pixels / total_pixels if total_pixels > 0 else 0
                    
                    is_checked = filled_ratio > 0.5  # More than half filled
                    
                    radio_buttons.append({
                        "x": x,
                        "y": y,
                        "radius": radius,
                        "is_checked": is_checked,
                        "filled_ratio": filled_ratio
                    })
        
        return radio_buttons
    
    def _detect_text_fields(self, 
                           horizontal_lines: np.ndarray, 
                           vertical_lines: np.ndarray, 
                           width: int, 
                           height: int) -> List[Dict[str, Any]]:
        """
        Detect text fields by analyzing line intersections.
        
        Args:
            horizontal_lines: Detected horizontal lines
            vertical_lines: Detected vertical lines
            width: Image width
            height: Image height
            
        Returns:
            List of detected text fields
        """
        text_fields = []
        
        if horizontal_lines is None or vertical_lines is None:
            return text_fields
        
        # Find horizontal line segments that likely represent text fields
        # (typically underlines for form fields)
        for h_line in horizontal_lines:
            x1, y1, x2, y2 = h_line[0]
            
            # Ensure line is horizontal
            if abs(y2 - y1) > 5:
                continue
                
            # Calculate line length
            line_length = abs(x2 - x1)
            
            # Skip short lines
            if line_length < 30:
                continue
            
            # Check vertical space above the line (should be empty for text field)
            space_height = 20  # Typical height for text field
            
            # Find if there's a boundary above this line (like another horizontal line)
            has_boundary_above = False
            boundary_y = 0
            
            for other_h_line in horizontal_lines:
                ox1, oy1, ox2, oy2 = other_h_line[0]
                
                # Check if it's above and within x-range
                if oy1 < y1 and abs(oy1 - y1) < 50:
                    # Check horizontal overlap
                    if (min(x2, ox2) - max(x1, ox1)) > 0:
                        has_boundary_above = True
                        boundary_y = max(boundary_y, oy1)
            
            # Define field boundaries
            field_x1 = min(x1, x2)
            field_x2 = max(x1, x2)
            field_y2 = y1  # Bottom of field is the line itself
            
            # Define top of field
            field_y1 = boundary_y if has_boundary_above else max(0, y1 - space_height)
            
            text_fields.append({
                "x": int(field_x1),
                "y": int(field_y1),
                "width": int(field_x2 - field_x1),
                "height": int(field_y2 - field_y1)
            })
        
        return text_fields
    
    def _detect_tables(self, 
                      horizontal_lines: np.ndarray, 
                      vertical_lines: np.ndarray, 
                      width: int, 
                      height: int) -> List[Dict[str, Any]]:
        """
        Detect tables in a form based on grid patterns.
        
        Args:
            horizontal_lines: Detected horizontal lines
            vertical_lines: Detected vertical lines
            width: Image width
            height: Image height
            
        Returns:
            List of detected tables with cell information
        """
        tables = []
        
        if horizontal_lines is None or vertical_lines is None:
            return tables
        
        # Group horizontal lines by y-coordinate proximity
        # (lines close to each other are likely part of the same table)
        h_groups = self._group_lines_by_position(horizontal_lines, axis=1, threshold=20)
        v_groups = self._group_lines_by_position(vertical_lines, axis=0, threshold=20)
        
        # Find potential tables
        for h_group in h_groups:
            if len(h_group) < 3:
                continue  # Need at least 3 horizontal lines for a table
                
            for v_group in v_groups:
                if len(v_group) < 2:
                    continue  # Need at least 2 vertical lines for a table
                    
                # Check if these groups form a table
                table = self._verify_table_structure(h_group, v_group)
                if table:
                    tables.append(table)
        
        return tables
    
    def _group_lines_by_position(self, lines: np.ndarray, axis: int, threshold: int) -> List[np.ndarray]:
        """
        Group lines that are close to each other along specified axis.
        
        Args:
            lines: Line coordinates
            axis: 0 for x-coordinate (vertical), 1 for y-coordinate (horizontal)
            threshold: Maximum distance to be considered in same group
            
        Returns:
            List of line groups
        """
        if lines is None or len(lines) == 0:
            return []
            
        # Extract relevant coordinates based on axis
        # For horizontal lines, we group by y; for vertical lines, we group by x
        coords = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            coord = y1 if axis == 1 else x1
            coords.append((coord, line))
        
        # Sort by coordinate
        coords.sort(key=lambda x: x[0])
        
        # Group lines with similar coordinates
        groups = []
        current_group = [coords[0][1]]
        current_pos = coords[0][0]
        
        for i in range(1, len(coords)):
            coord, line = coords[i]
            
            if abs(coord - current_pos) <= threshold:
                current_group.append(line)
            else:
                groups.append(np.array(current_group))
                current_group = [line]
                current_pos = coord
        
        if current_group:
            groups.append(np.array(current_group))
        
        return groups
    
    def _verify_table_structure(self, horizontal_lines: np.ndarray, vertical_lines: np.ndarray) -> Dict[str, Any]:
        """
        Verify if the given lines form a table structure.
        
        Args:
            horizontal_lines: Group of horizontal lines
            vertical_lines: Group of vertical lines
            
        Returns:
            Table information if valid, None otherwise
        """
        # Find table boundaries
        h_x_min = min(min(x1, x2) for line in horizontal_lines for x1, y1, x2, y2 in [line[0]])
        h_x_max = max(max(x1, x2) for line in horizontal_lines for x1, y1, x2, y2 in [line[0]])
        h_y_min = min(min(y1, y2) for line in horizontal_lines for x1, y1, x2, y2 in [line[0]])
        h_y_max = max(max(y1, y2) for line in horizontal_lines for x1, y1, x2, y2 in [line[0]])
        
        v_x_min = min(min(x1, x2) for line in vertical_lines for x1, y1, x2, y2 in [line[0]])
        v_x_max = max(max(x1, x2) for line in vertical_lines for x1, y1, x2, y2 in [line[0]])
        v_y_min = min(min(y1, y2) for line in vertical_lines for x1, y1, x2, y2 in [line[0]])
        v_y_max = max(max(y1, y2) for line in vertical_lines for x1, y1, x2, y2 in [line[0]])
        
        # Check if horizontal and vertical lines overlap sufficiently
        x_overlap = min(h_x_max, v_x_max) - max(h_x_min, v_x_min)
        y_overlap = min(h_y_max, v_y_max) - max(h_y_min, v_y_min)
        
        if x_overlap <= 0 or y_overlap <= 0:
            return None
        
        # Calculate table dimensions
        table_x = max(h_x_min, v_x_min)
        table_y = max(h_y_min, v_y_min)
        table_width = min(h_x_max, v_x_max) - table_x
        table_height = min(h_y_max, v_y_max) - table_y
        
        # Count rows and columns
        row_positions = sorted(set(min(y1, y2) for line in horizontal_lines for x1, y1, x2, y2 in [line[0]]))
        col_positions = sorted(set(min(x1, x2) for line in vertical_lines for x1, y1, x2, y2 in [line[0]]))
        
        rows = len(row_positions) - 1
        cols = len(col_positions) - 1
        
        # Construct table info only if we have multiple rows and columns
        if rows >= 1 and cols >= 1:
            # Calculate cells
            cells = []
            
            for i in range(rows):
                row_y1 = row_positions[i]
                row_y2 = row_positions[i+1]
                
                for j in range(cols):
                    col_x1 = col_positions[j]
                    col_x2 = col_positions[j+1]
                    
                    cells.append({
                        "row": i,
                        "col": j,
                        "x": int(col_x1),
                        "y": int(row_y1),
                        "width": int(col_x2 - col_x1),
                        "height": int(row_y2 - row_y1)
                    })
            
            return {
                "x": int(table_x),
                "y": int(table_y),
                "width": int(table_width),
                "height": int(table_height),
                "rows": rows,
                "cols": cols,
                "cells": cells
            }
        
        return None
    
    def _associate_text_with_elements(self, 
                                     form_elements: Dict[str, List[Dict[str, Any]]], 
                                     text_regions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Associate OCR text with detected form elements.
        
        Args:
            form_elements: Detected form elements
            text_regions: OCR text regions
            
        Returns:
            Dictionary with form fields and associated labels/values
        """
        result = {
            "fields": [],
            "tables": [],
            "checkboxes": [],
            "radio_groups": []
        }
        
        # Group checkboxes into radio button groups
        radio_groups = self._group_radio_buttons(form_elements["radio_buttons"])
        
        # Process text fields
        for field in form_elements["text_fields"]:
            field_info = {
                "type": "text_field",
                "bounds": {
                    "x": field["x"],
                    "y": field["y"],
                    "width": field["width"],
                    "height": field["height"]
                },
                "label": "",
                "value": ""
            }
            
            # Find label (text region above or to the left of field)
            field_info["label"] = self._find_field_label(field, text_regions)
            
            # Find value (text within field bounds)
            field_info["value"] = self._find_field_value(field, text_regions)
            
            result["fields"].append(field_info)
        
        # Process checkboxes
        for checkbox in form_elements["checkboxes"]:
            checkbox_info = {
                "type": "checkbox",
                "bounds": {
                    "x": checkbox["x"],
                    "y": checkbox["y"],
                    "width": checkbox["width"],
                    "height": checkbox["height"]
                },
                "label": "",
                "is_checked": checkbox["is_checked"]
            }
            
            # Find label (text region to the right or below checkbox)
            checkbox_info["label"] = self._find_checkbox_label(checkbox, text_regions)
            
            result["checkboxes"].append(checkbox_info)
        
        # Process radio button groups
        for group_idx, group in enumerate(radio_groups):
            radio_group_info = {
                "type": "radio_group",
                "group_id": f"group_{group_idx}",
                "options": []
            }
            
            for button in group:
                option_info = {
                    "bounds": {
                        "x": button["x"],
                        "y": button["y"],
                        "radius": button["radius"]
                    },
                    "label": self._find_checkbox_label(
                        {"x": button["x"] - button["radius"],
                         "y": button["y"] - button["radius"],
                         "width": 2 * button["radius"],
                         "height": 2 * button["radius"]},
                        text_regions
                    ),
                    "is_selected": button["is_checked"]
                }
                
                radio_group_info["options"].append(option_info)
            
            result["radio_groups"].append(radio_group_info)
        
        # Process tables
        for table in form_elements["tables"]:
            table_info = {
                "bounds": {
                    "x": table["x"],
                    "y": table["y"],
                    "width": table["width"],
                    "height": table["height"]
                },
                "rows": table["rows"],
                "cols": table["cols"],
                "cells": []
            }
            
            for cell in table["cells"]:
                cell_info = {
                    "row": cell["row"],
                    "col": cell["col"],
                    "bounds": {
                        "x": cell["x"],
                        "y": cell["y"],
                        "width": cell["width"],
                        "height": cell["height"]
                    },
                    "text": self._find_cell_text(cell, text_regions)
                }
                
                table_info["cells"].append(cell_info)
            
            result["tables"].append(table_info)
        
        return result
    
    def _group_radio_buttons(self, radio_buttons: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Group radio buttons that are likely part of the same group.
        
        Args:
            radio_buttons: List of detected radio buttons
            
        Returns:
            List of radio button groups
        """
        if not radio_buttons:
            return []
        
        # Sort buttons by y-coordinate first
        radio_buttons.sort(key=lambda x: x["y"])
        
        groups = []
        current_group = [radio_buttons[0]]
        current_y = radio_buttons[0]["y"]
        
        # Group buttons that are horizontally aligned (similar y-coordinate)
        for button in radio_buttons[1:]:
            if abs(button["y"] - current_y) <= 20:  # Allow small vertical variation
                current_group.append(button)
            else:
                groups.append(current_group)
                current_group = [button]
                current_y = button["y"]
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _find_field_label(self, field: Dict[str, Any], text_regions: List[Dict[str, Any]]) -> str:
        """
        Find the label for a form field by looking at nearby text.
        
        Args:
            field: Field coordinates
            text_regions: OCR text regions
            
        Returns:
            Extracted label text
        """
        field_x, field_y = field["x"], field["y"]
        field_width, field_height = field["width"], field["height"]
        
        closest_label = ""
        min_distance = float('inf')
        
        for region in text_regions:
            # Get region center
            box = region.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
            
            # Calculate region boundaries
            min_x = min(p[0] for p in box)
            min_y = min(p[1] for p in box)
            max_x = max(p[0] for p in box)
            max_y = max(p[1] for p in box)
            
            region_width = max_x - min_x
            region_height = max_y - min_y
            
            # Skip if region is too large (likely not a label)
            if region_width > 300 or region_height > 50:
                continue
            
            # Calculate distance metrics for different positions
            # Left of field
            left_distance = abs(field_x - max_x)
            left_aligned = min_y <= field_y + field_height and max_y >= field_y
            
            # Above field
            above_distance = abs(field_y - max_y)
            above_aligned = min_x <= field_x + field_width and max_x >= field_x
            
            # Choose closest aligned region
            if left_aligned and left_distance < 100:
                if left_distance < min_distance:
                    min_distance = left_distance
                    closest_label = region.get("text", "")
            elif above_aligned and above_distance < 50:
                if above_distance < min_distance:
                    min_distance = above_distance
                    closest_label = region.get("text", "")
        
        # Clean up the label (remove trailing colons, etc.)
        if closest_label:
            closest_label = re.sub(r'[:*]+$', '', closest_label).strip()
        
        return closest_label
    
    def _find_field_value(self, field: Dict[str, Any], text_regions: List[Dict[str, Any]]) -> str:
        """
        Find text value within a form field's boundaries.
        
        Args:
            field: Field coordinates
            text_regions: OCR text regions
            
        Returns:
            Extracted value text
        """
        field_x, field_y = field["x"], field["y"]
        field_width, field_height = field["width"], field["height"]
        
        field_value = ""
        
        for region in text_regions:
            # Get region center
            box = region.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
            
            # Calculate region boundaries
            min_x = min(p[0] for p in box)
            min_y = min(p[1] for p in box)
            max_x = max(p[0] for p in box)
            max_y = max(p[1] for p in box)
            
            # Calculate overlap
            overlap_x = min(field_x + field_width, max_x) - max(field_x, min_x)
            overlap_y = min(field_y + field_height, max_y) - max(field_y, min_y)
            
            # Check if region is inside field
            if overlap_x > 0 and overlap_y > 0:
                # Calculate overlap area
                overlap_area = overlap_x * overlap_y
                region_area = (max_x - min_x) * (max_y - min_y)
                
                # If significant portion of the region is inside field
                if overlap_area / region_area > 0.5:
                    field_value = region.get("text", "")
                    break
        
        return field_value
    
    def _find_checkbox_label(self, checkbox: Dict[str, Any], text_regions: List[Dict[str, Any]]) -> str:
        """
        Find label text for a checkbox or radio button.
        
        Args:
            checkbox: Checkbox coordinates
            text_regions: OCR text regions
            
        Returns:
            Extracted label text
        """
        checkbox_x, checkbox_y = checkbox["x"], checkbox["y"]
        checkbox_width, checkbox_height = checkbox["width"], checkbox["height"]
        
        closest_label = ""
        min_distance = float('inf')
        
        for region in text_regions:
            # Get region center
            box = region.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
            
            # Calculate region boundaries
            min_x = min(p[0] for p in box)
            min_y = min(p[1] for p in box)
            max_x = max(p[0] for p in box)
            max_y = max(p[1] for p in box)
            
            # Calculate horizontal distance to the right of checkbox
            right_distance = min_x - (checkbox_x + checkbox_width)
            
            # Check if region is to the right and aligned with checkbox
            if right_distance > 0 and right_distance < 100:
                if min_y <= checkbox_y + checkbox_height and max_y >= checkbox_y:
                    if right_distance < min_distance:
                        min_distance = right_distance
                        closest_label = region.get("text", "")
        
        return closest_label
    
    def _find_cell_text(self, cell: Dict[str, Any], text_regions: List[Dict[str, Any]]) -> str:
        """
        Find text within a table cell.
        
        Args:
            cell: Cell coordinates
            text_regions: OCR text regions
            
        Returns:
            Extracted cell text
        """
        cell_x, cell_y = cell["x"], cell["y"]
        cell_width, cell_height = cell["width"], cell["height"]
        
        cell_text = ""
        cell_regions = []
        
        for region in text_regions:
            # Get region center
            box = region.get("box", [[0, 0], [0, 0], [0, 0], [0, 0]])
            
            # Calculate region boundaries
            min_x = min(p[0] for p in box)
            min_y = min(p[1] for p in box)
            max_x = max(p[0] for p in box)
            max_y = max(p[1] for p in box)
            
            # Calculate center of region
            region_center_x = (min_x + max_x) / 2
            region_center_y = (min_y + max_y) / 2
            
            # Check if region center is inside cell
            if (cell_x <= region_center_x <= cell_x + cell_width and
                cell_y <= region_center_y <= cell_y + cell_height):
                cell_regions.append((region.get("text", ""), min_y))
        
        # Sort regions by y-coordinate to maintain reading order
        cell_regions.sort(key=lambda x: x[1])
        
        # Combine text from all regions in the cell
        cell_text = " ".join(text for text, _ in cell_regions)
        
        return cell_text
    
    def _structure_fields(self, fields: Dict[str, Any]) -> Dict[str, Any]:
        """
        Post-process and structure extracted form fields.
        
        Args:
            fields: Raw field extractions
            
        Returns:
            Structured form data
        """
        # Convert to dictionary structure
        structured_data = {
            "form_fields": {},
            "tables": [],
            "checkboxes": {},
            "radio_groups": {}
        }
        
        # Process text fields
        for field in fields["fields"]:
            label = field["label"] or f"Field_{len(structured_data['form_fields'])}"
            structured_data["form_fields"][label] = field["value"]
        
        # Process checkboxes
        for checkbox in fields["checkboxes"]:
            label = checkbox["label"] or f"Checkbox_{len(structured_data['checkboxes'])}"
            structured_data["checkboxes"][label] = checkbox["is_checked"]
        
        # Process radio groups
        for group in fields["radio_groups"]:
            group_id = group["group_id"]
            options = {}
            selected = None
            
            for option in group["options"]:
                label = option["label"] or f"Option_{len(options)}"
                options[label] = option["is_selected"]
                
                if option["is_selected"]:
                    selected = label
            
            structured_data["radio_groups"][group_id] = {
                "options": options,
                "selected": selected
            }
        
        # Process tables
        for table in fields["tables"]:
            table_data = []
            
            # Get number of rows and columns
            rows = table["rows"]
            cols = table["cols"]
            
            # Initialize table data structure
            for _ in range(rows):
                table_data.append([""] * cols)
            
            # Fill in cell data
            for cell in table["cells"]:
                row = cell["row"]
                col = cell["col"]
                
                if 0 <= row < rows and 0 <= col < cols:
                    table_data[row][col] = cell["text"]
            
            structured_data["tables"].append(table_data)
        
        return structured_data
    
    def process_form(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Process a form document image.
        
        Args:
            image_path: Path to the form image
            
        Returns:
            Structured form data
        """
        if self.ocr_processor is None:
            return {"error": "OCR processor not provided"}
            
        try:
            # Load image
            image = cv2.imread(str(image_path))
            if image is None:
                return {"error": f"Failed to load image: {image_path}"}
            
            # Run OCR to get text regions
            ocr_result = self.ocr_processor.process_image(image_path)
            text_regions = ocr_result.get("results", [])
            
            # Extract form fields
            form_data = self.extract_fields(image, text_regions)
            
            # Structure the results
            structured_data = self._structure_fields(form_data)
            
            return {
                "image_path": str(image_path),
                "form_data": structured_data
            }
            
        except Exception as e:
            return {"error": f"Failed to process form: {str(e)}"}
    
    def extract_form_fields(self, document_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Extract form fields from a document.
        
        Args:
            document_path: Path to the document
            
        Returns:
            Dictionary with extracted form fields
        """
        # Determine if it's an image or PDF
        document_path = str(document_path)
        
        if document_path.lower().endswith(('.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp')):
            # Process as image
            return self.process_form(document_path)
        elif document_path.lower().endswith('.pdf'):
            # Process as PDF
            try:
                import fitz  # PyMuPDF
                
                # Open PDF
                doc = fitz.open(document_path)
                
                # Process first page
                if len(doc) > 0:
                    # Convert first page to image
                    page = doc[0]
                    pix = page.get_pixmap(dpi=300)
                    
                    # Save as temporary image
                    temp_path = f"{os.path.splitext(document_path)[0]}_temp.png"
                    pix.save(temp_path)
                    
                    try:
                        # Process the image
                        result = self.process_form(temp_path)
                        return result
                    finally:
                        # Clean up temp file
                        if os.path.exists(temp_path):
                            os.unlink(temp_path)
                
                return {"error": "Failed to process PDF form"}
                
            except ImportError:
                return {"error": "PyMuPDF (fitz) library required for PDF processing"}
            except Exception as e:
                return {"error": f"Failed to process PDF form: {str(e)}"}
        else:
            return {"error": "Unsupported document format"}
