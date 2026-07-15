"""Optional Pascal VOC XML export (bounding boxes; polygons approximated when available)."""

from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

from auto_annotation.config import AppConfig


def _indent(elem, level=0):
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            _indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def export_voc(image_records: list[dict], cfg: AppConfig) -> Path:
    """Export Pascal VOC XML with bboxes. Masks are NOT native VOC — bbox only."""
    cfg.voc_directory.mkdir(parents=True, exist_ok=True)
    readme = cfg.voc_directory / "README.txt"
    readme.write_text(
        "Pascal VOC XML exports bounding boxes only.\n"
        "For full instance masks use COCO or YOLO segmentation exports.\n",
        encoding="utf-8",
    )

    for rec in image_records:
        if rec.get("status") != "ok":
            continue
        w, h = rec["width"], rec["height"]
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = "images"
        ET.SubElement(root, "filename").text = rec["file_name"]
        size = ET.SubElement(root, "size")
        ET.SubElement(size, "width").text = str(w)
        ET.SubElement(size, "height").text = str(h)
        ET.SubElement(size, "depth").text = "3"

        for inst in rec["instances"]:
            obj = ET.SubElement(root, "object")
            ET.SubElement(obj, "name").text = inst["class_name"]
            ET.SubElement(obj, "pose").text = "Unspecified"
            ET.SubElement(obj, "truncated").text = "0"
            ET.SubElement(obj, "difficult").text = "0"
            bnd = ET.SubElement(obj, "bndbox")
            x1, y1, x2, y2 = inst["bbox_xyxy"]
            ET.SubElement(bnd, "xmin").text = str(int(x1))
            ET.SubElement(bnd, "ymin").text = str(int(y1))
            ET.SubElement(bnd, "xmax").text = str(int(x2))
            ET.SubElement(bnd, "ymax").text = str(int(y2))

        _indent(root)
        out = cfg.voc_directory / f"{Path(rec['file_name']).stem}.xml"
        ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)

    return cfg.voc_directory
