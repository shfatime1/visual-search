import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from .images import prepare_image


def load_catalog(data_dir: Path):
    with (data_dir / "products.csv").open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"id", "name", "brand", "category", "subcategory", "price_usd", "image_file"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("products.csv is missing required columns")
        rows = list(reader)
    report = {"total_rows": len(rows), "missing_values": {}, "invalid_rows": [], "duplicate_rows": [],
              "duplicate_ids": [], "duplicate_image_paths": [], "duplicate_image_content": []}
    valid, seen_rows, seen_ids = [], set(), set()
    hashes, paths = defaultdict(list), defaultdict(list)
    for number, row in enumerate(rows, 2):
        for key, value in row.items():
            if not value or not value.strip():
                report["missing_values"].setdefault(str(key), []).append(number)
        signature = json.dumps(row, sort_keys=True)
        if signature in seen_rows:
            report["duplicate_rows"].append(number)
        seen_rows.add(signature)
        try:
            pid = int(row["id"])
            if pid in seen_ids:
                report["duplicate_ids"].append(pid)
                raise ValueError("Duplicate ID; only first valid occurrence is indexed")
            if not all(row[k] and row[k].strip() for k in ("name", "category", "subcategory", "image_file")):
                raise ValueError("Missing required product field")
            price = float(row["price_usd"])
            if not math.isfinite(price) or price < 0:
                raise ValueError("Invalid price")
            path = (data_dir / row["image_file"]).resolve()
            if not path.is_relative_to((data_dir / "images").resolve()) or path.parent != (data_dir / "images").resolve():
                raise ValueError("Image must be a direct child of data/images")
            content = path.read_bytes()
            prepare_image(content)
            hashes[hashlib.sha256(content).hexdigest()].append(pid)
            paths[str(path)].append(pid)
            valid.append({"id": pid, "title": row["name"], "brand": row["brand"], "category": row["category"],
                          "subcategory": row["subcategory"], "price_usd": price,
                          "image_url": "/images/" + path.name, "_path": path})
            seen_ids.add(pid)
        except (ValueError, TypeError, OSError) as exc:
            report["invalid_rows"].append({"csv_line": number, "reason": str(exc)})
    report["valid_products"] = len(valid)
    report["image_files"] = len(list((data_dir / "images").glob("*")))
    report["duplicate_image_content"] = [ids for ids in hashes.values() if len(ids) > 1]
    report["duplicate_image_paths"] = [ids for ids in paths.values() if len(ids) > 1]
    report["unreferenced_images"] = [p.name for p in (data_dir / "images").glob("*") if str(p.resolve()) not in paths]
    return valid, report


def public_product(product):
    return {k: v for k, v in product.items() if not k.startswith("_")}


def fingerprint(products):
    h = hashlib.sha256()
    for p in products:
        h.update(json.dumps(public_product(p), sort_keys=True).encode())
        h.update(p["_path"].read_bytes())
    return h.hexdigest()
