import json
import zipfile
from pathlib import Path
from logger import get_logger

logger = get_logger("utils")

def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def zip_dir(src_dir, output_zip):
    src = Path(src_dir)
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(src.iterdir()):
            if file.is_file():
                z.write(file, arcname=file.name)
    logger.info(f"Created zip: {output_zip}")
