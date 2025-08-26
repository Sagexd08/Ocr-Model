from typing import Dict, Any, List, Optional
import os
import json
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

from ...types import Document
from ...utils.logging import get_logger, log_execution_time

logger = get_logger(__name__)

class ExportFormat(BaseModel):
    """
    Configuration for export formats
    """
    format_type: str = Field(..., description="Format type (json, csv, pdf, etc.)")
    include_confidence: bool = Field(False, description="Include confidence scores")
    include_bbox: bool = Field(False, description="Include bounding box coordinates")
    include_metadata: bool = Field(True, description="Include document metadata")
    structure_preserving: bool = Field(True, description="Preserve document structure")
    schema: Optional[Dict[str, Any]] = Field(None, description="Output schema definition")


class Exporter:
    """
    Document exporter that converts processed OCR results to various output formats.
    Supports JSON, CSV, XML, PDF with annotations, and custom formats.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.output_dir = self.config.get("output_dir", "output")
        self.default_format = self.config.get("default_format", "json")
        
        # Configure export formats from config
        self.export_formats = {}
        formats_config = self.config.get("formats", {})
        for format_name, format_config in formats_config.items():
            self.export_formats[format_name] = ExportFormat(**format_config)
        
        # Ensure at least one default format exists
        if not self.export_formats:
            self.export_formats["json"] = ExportFormat(
                format_type="json",
                include_confidence=True,
                include_bbox=True,
                include_metadata=True,
                structure_preserving=True
            )
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
    
    @log_execution_time
    def export(self, document: Document, format_name: Optional[str] = None) -> Dict[str, str]:
        """
        Export document to specified format.
        
        Args:
            document: Processed document to export
            format_name: Name of format to use (defaults to default_format)
            
        Returns:
            Dict with file paths for exported documents
        """
        format_name = format_name or self.default_format
        if format_name not in self.export_formats:
            logger.warning(f"Format {format_name} not found, using {self.default_format}")
            format_name = self.default_format
            
        export_format = self.export_formats[format_name]
        
        logger.info(f"Exporting document {document.id} to {format_name} format")
        
        try:
            # Generate export filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_base = f"{document.id}_{timestamp}"
            
            # Export based on format type
            if export_format.format_type == "json":
                filepath = self._export_json(document, filename_base, export_format)
            elif export_format.format_type == "csv":
                filepath = self._export_csv(document, filename_base, export_format)
            elif export_format.format_type == "xml":
                filepath = self._export_xml(document, filename_base, export_format)
            elif export_format.format_type == "pdf":
                filepath = self._export_pdf(document, filename_base, export_format)
            elif export_format.format_type == "txt":
                filepath = self._export_txt(document, filename_base, export_format)
            else:
                logger.error(f"Unsupported export format: {export_format.format_type}")
                return {}
                
            return {format_name: filepath}
            
        except Exception as e:
            logger.error(f"Error exporting document {document.id}: {str(e)}")
            return {}
    
    def _export_json(self, document: Document, filename_base: str, format_config: ExportFormat) -> str:
        """Export document to JSON format"""
        filepath = os.path.join(self.output_dir, f"{filename_base}.json")
        
        # Convert document to dict representation
        doc_dict = document.dict()
        
        # Apply format configuration
        if not format_config.include_confidence:
            self._remove_confidence_scores(doc_dict)
            
        if not format_config.include_bbox:
            self._remove_bboxes(doc_dict)
            
        if not format_config.include_metadata:
            doc_dict.pop("metadata", None)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(doc_dict, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Exported JSON to {filepath}")
        return filepath
    
    def _export_csv(self, document: Document, filename_base: str, format_config: ExportFormat) -> str:
        """Export document to CSV format"""
        import csv
        
        filepath = os.path.join(self.output_dir, f"{filename_base}.csv")
        
        # For CSV, we flatten the document structure
        rows = []
        
        # Add header
        headers = ["page_num", "region_type", "text"]
        if format_config.include_bbox:
            headers.extend(["x1", "y1", "x2", "y2"])
        if format_config.include_confidence:
            headers.append("confidence")
            
        rows.append(headers)
        
        # Add document content
        for page in document.pages:
            # Add regions
            for region in page.regions:
                row = [
                    page.page_num,
                    region.type,
                    region.text
                ]
                
                if format_config.include_bbox:
                    row.extend([
                        region.bbox.x1,
                        region.bbox.y1,
                        region.bbox.x2,
                        region.bbox.y2
                    ])
                    
                if format_config.include_confidence:
                    row.append(region.confidence)
                    
                rows.append(row)
                
            # Add tables
            for table in page.tables:
                for row_idx, cells in enumerate(table.cells):
                    for col_idx, cell in enumerate(cells):
                        if not cell or not cell.text:
                            continue
                            
                        row = [
                            page.page_num,
                            f"table_cell_{table.id}_r{row_idx}_c{col_idx}",
                            cell.text
                        ]
                        
                        if format_config.include_bbox:
                            row.extend([
                                cell.bbox.x1,
                                cell.bbox.y1,
                                cell.bbox.x2,
                                cell.bbox.y2
                            ])
                            
                        if format_config.include_confidence:
                            row.append(cell.confidence)
                            
                        rows.append(row)
        
        # Write to file
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(rows)
            
        logger.info(f"Exported CSV to {filepath}")
        return filepath
    
    def _export_xml(self, document: Document, filename_base: str, format_config: ExportFormat) -> str:
        """Export document to XML format"""
        import xml.etree.ElementTree as ET
        from xml.dom import minidom
        
        filepath = os.path.join(self.output_dir, f"{filename_base}.xml")
        
        # Create XML structure
        root = ET.Element("Document")
        root.set("id", str(document.id))
        
        # Add metadata if configured
        if format_config.include_metadata and document.metadata:
            metadata_elem = ET.SubElement(root, "Metadata")
            for key, value in document.metadata.items():
                if isinstance(value, dict):
                    meta_section = ET.SubElement(metadata_elem, key)
                    for k, v in value.items():
                        meta_item = ET.SubElement(meta_section, k)
                        meta_item.text = str(v)
                else:
                    meta_item = ET.SubElement(metadata_elem, key)
                    meta_item.text = str(value)
        
        # Add pages
        pages_elem = ET.SubElement(root, "Pages")
        for page in document.pages:
            page_elem = ET.SubElement(pages_elem, "Page")
            page_elem.set("number", str(page.page_num))
            
            # Add regions
            regions_elem = ET.SubElement(page_elem, "Regions")
            for region in page.regions:
                region_elem = ET.SubElement(regions_elem, "Region")
                region_elem.set("type", region.type)
                region_elem.set("id", str(region.id))
                
                # Add bounding box if configured
                if format_config.include_bbox:
                    bbox_elem = ET.SubElement(region_elem, "BoundingBox")
                    bbox_elem.set("x1", str(region.bbox.x1))
                    bbox_elem.set("y1", str(region.bbox.y1))
                    bbox_elem.set("x2", str(region.bbox.x2))
                    bbox_elem.set("y2", str(region.bbox.y2))
                
                # Add confidence if configured
                if format_config.include_confidence:
                    region_elem.set("confidence", str(region.confidence))
                
                # Add text content
                text_elem = ET.SubElement(region_elem, "Text")
                text_elem.text = region.text
            
            # Add tables
            tables_elem = ET.SubElement(page_elem, "Tables")
            for table in page.tables:
                table_elem = ET.SubElement(tables_elem, "Table")
                table_elem.set("id", str(table.id))
                
                rows_elem = ET.SubElement(table_elem, "Rows")
                for row_idx, cells in enumerate(table.cells):
                    row_elem = ET.SubElement(rows_elem, "Row")
                    row_elem.set("index", str(row_idx))
                    
                    for col_idx, cell in enumerate(cells):
                        if cell:
                            cell_elem = ET.SubElement(row_elem, "Cell")
                            cell_elem.set("col", str(col_idx))
                            
                            if format_config.include_bbox:
                                bbox_elem = ET.SubElement(cell_elem, "BoundingBox")
                                bbox_elem.set("x1", str(cell.bbox.x1))
                                bbox_elem.set("y1", str(cell.bbox.y1))
                                bbox_elem.set("x2", str(cell.bbox.x2))
                                bbox_elem.set("y2", str(cell.bbox.y2))
                            
                            if format_config.include_confidence:
                                cell_elem.set("confidence", str(cell.confidence))
                            
                            text_elem = ET.SubElement(cell_elem, "Text")
                            text_elem.text = cell.text
        
        # Write to file with pretty formatting
        xml_str = ET.tostring(root, encoding='utf-8')
        parsed_xml = minidom.parseString(xml_str)
        pretty_xml = parsed_xml.toprettyxml(indent="  ")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(pretty_xml)
            
        logger.info(f"Exported XML to {filepath}")
        return filepath
    
    def _export_pdf(self, document: Document, filename_base: str, format_config: ExportFormat) -> str:
        """
        Export to PDF with annotations.
        Note: This would require additional libraries like reportlab or PyMuPDF.
        """
        filepath = os.path.join(self.output_dir, f"{filename_base}_annotated.pdf")
        
        # For demonstration purposes only
        # In a production system, this would create a PDF with the original
        # document images and overlay the extracted text and regions
        
        logger.warning("PDF export is not fully implemented")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("PDF export not fully implemented")
            
        logger.info(f"Exported PDF placeholder to {filepath}")
        return filepath
    
    def _export_txt(self, document: Document, filename_base: str, format_config: ExportFormat) -> str:
        """Export document to plain text format"""
        filepath = os.path.join(self.output_dir, f"{filename_base}.txt")
        
        text_content = []
        
        # Add document title if available
        if "title" in document.metadata:
            text_content.append(f"# {document.metadata['title']}")
            text_content.append("")
        
        # Add document content
        for page in document.pages:
            text_content.append(f"=== Page {page.page_num} ===")
            text_content.append("")
            
            # Add regions, preserving structure
            for region in page.regions:
                if format_config.structure_preserving:
                    if region.type == "heading":
                        text_content.append(f"## {region.text}")
                    elif region.type == "subheading":
                        text_content.append(f"### {region.text}")
                    elif region.type == "paragraph":
                        text_content.append(region.text)
                    elif region.type == "list":
                        for line in region.text.split('\n'):
                            text_content.append(f"* {line.strip()}")
                    else:
                        text_content.append(region.text)
                else:
                    text_content.append(region.text)
                
                text_content.append("")
            
            # Add tables as text
            for table_idx, table in enumerate(page.tables):
                text_content.append(f"Table {table_idx + 1}:")
                
                # Calculate column widths for better formatting
                col_widths = [0] * (max(len(row) for row in table.cells) if table.cells else 0)
                for row in table.cells:
                    for i, cell in enumerate(row):
                        if cell and cell.text:
                            col_widths[i] = max(col_widths[i], len(cell.text))
                
                # Add table content
                for row in table.cells:
                    row_text = "|"
                    for i, cell in enumerate(row):
                        if i < len(col_widths):
                            text = cell.text if cell and cell.text else ""
                            row_text += f" {text.ljust(col_widths[i])} |"
                    text_content.append(row_text)
                
                text_content.append("")
            
            text_content.append("")
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(text_content))
            
        logger.info(f"Exported TXT to {filepath}")
        return filepath
    
    def _remove_confidence_scores(self, doc_dict: Dict[str, Any]):
        """Remove confidence scores from document dict"""
        # Remove document level confidence
        doc_dict.pop("confidence", None)
        
        # Remove page level confidences
        for page in doc_dict.get("pages", []):
            page.pop("confidence", None)
            
            # Remove region confidences
            for region in page.get("regions", []):
                region.pop("confidence", None)
                
                # Remove token confidences
                for token in region.get("tokens", []):
                    token.pop("confidence", None)
            
            # Remove table confidences
            for table in page.get("tables", []):
                table.pop("confidence", None)
                
                # Remove cell confidences
                for row in table.get("cells", []):
                    for cell in row:
                        if cell:
                            cell.pop("confidence", None)
    
    def _remove_bboxes(self, doc_dict: Dict[str, Any]):
        """Remove bounding boxes from document dict"""
        # Remove page bbox
        for page in doc_dict.get("pages", []):
            page.pop("bbox", None)
            
            # Remove region bboxes
            for region in page.get("regions", []):
                region.pop("bbox", None)
                
                # Remove token bboxes
                for token in region.get("tokens", []):
                    token.pop("bbox", None)
            
            # Remove table bboxes
            for table in page.get("tables", []):
                table.pop("bbox", None)
                
                # Remove cell bboxes
                for row in table.get("cells", []):
                    for cell in row:
                        if cell:
                            cell.pop("bbox", None)
