from label_studio_sdk import Client
from utils.logger import get_logger
from utils.constants import DEFECT_CLASSES
import datetime
import os
import json

logger = get_logger(__name__)

# Reverse mapping: label_name -> category_id
label_to_id = {v: k for k, v in DEFECT_CLASSES.items()}

def process_task(task):
    task_id = task['id']
    image_url = task['data']['image']
    annotations = task.get("annotations", [])
    if len(annotations) == 0:
        logger.warning(f"Task {task_id} has no annotations")
        return {}

    final_annotation = annotations[-1]
    if final_annotation.get("was_cancelled", False):
      logger.warning(f"Task {task_id} was cancelled")
      return {}
    
    results = final_annotation.get('result', [])

    orig_w, orig_h = None, None
    # case 1: positive sample (annotation result exists)
    if len(results) > 0:
        orig_w = results[0].get("original_width")
        orig_h = results[0].get("original_height")

    # case 2: negative sample but prediction exists
    if (orig_w is None or orig_h is None):
        pred = final_annotation.get("prediction")
        if pred and pred.get("result"):
            orig_w = pred["result"][0].get("original_width")
            orig_h = pred["result"][0].get("original_height")
    annos = []
    for res in results:
        if "rectanglelabels" not in res["value"]:
            continue

        label_name = res["value"]["rectanglelabels"][0]

        if label_name not in label_to_id:
            print(f"WARNING: '{label_name}' not in predefined classes, skipping.")
            continue

        cat_id = label_to_id[label_name]

        x_pct = res["value"]["x"]
        y_pct = res["value"]["y"]
        w_pct = res["value"]["width"]
        h_pct = res["value"]["height"]

        # convert percent → pixels
        bbox = [
            (x_pct / 100) * orig_w,
            (y_pct / 100) * orig_h,
            (w_pct / 100) * orig_w,
            (h_pct / 100) * orig_h
        ]

        area = bbox[2] * bbox[3]

        annos.append({
            "label": label_name,
            "bbox": bbox,
            "area": area,
            "iscrowd": 0
        })
    return {
        "image_url": image_url,
        "width": orig_w,
        "height": orig_h,
        "annos": annos
    }

def pull_data_from_label_studio(url: str = "https://labelstudio.laka.ai", api_key: str = "31d6593ddbaf44e3c013bd0bfe11619326b352a0", project_id: int = 22, start: str = "2026-04-19", end: str = "2026-04-21", is_pull_old_dataset: bool = False):
    ls = Client(url, api_key)
    project = ls.get_project(project_id)
    tasks = project.get_tasks()
    logger.info(f"Pulled {len(tasks)} tasks from Label Studio")
    
    coco_output_format = {
        "info": {},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": []
    }
    # Parse start/end into date objects for proper comparison
    start_date = datetime.datetime.strptime(start, "%Y-%m-%d").date()
    end_date = datetime.datetime.strptime(end, "%Y-%m-%d").date()

    for task in tasks:
        created_at_str = task['created_at']  # e.g. 2026-04-21T06:15:21.453940Z
        created_date = datetime.datetime.fromisoformat(created_at_str.replace('Z', '+00:00')).date()
        if is_pull_old_dataset or (start_date <= created_date <= end_date):
            sample = process_task(task)
            if sample:
                image_id = len(coco_output_format['images']) + 1
                coco_output_format['images'].append({
                    "id": image_id,
                    "file_name": sample['image_url'],
                    "width": sample['width'],
                    "height": sample['height']
                })
                for anno in sample['annos']:
                    coco_output_format['annotations'].append({
                        "id": len(coco_output_format['annotations']) + 1,
                        "image_id": image_id,
                        "category_id": label_to_id.get(anno['label'], -1),
                        "bbox": anno['bbox'],
                        "area": anno['area'],
                        "iscrowd": anno['iscrowd']
                    })

    for category in DEFECT_CLASSES.items():
        coco_output_format['categories'].append({
            "id": category[0],
            "name": category[1],
            "supercategory": "defect"
        })
    json_path = f"tmp/annotations_{'old' if is_pull_old_dataset else 'new'}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(coco_output_format, f, indent=2)
    logger.info(f"Saved COCO dataset to {json_path}")
    return json_path

if __name__ == "__main__":
    pull_data_from_label_studio(is_pull_old_dataset=True)