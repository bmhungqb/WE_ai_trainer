"""
Update model_version in task JSON based on image path.
If image path contains a folder name (e.g. TPWL, TPRL),
appends it as suffix: final_preannotation -> final_preannotation_TPWL
"""

import json
import argparse
import re


def update_model_versions(input_path: str, output_path: str = None):
    with open(input_path, "r") as f:
        tasks = json.load(f)

    for task in tasks:
        image_url = task.get("data", {}).get("image", "")
        # Extract folder name from path like gs://bucket/TPWL/image.png
        match = re.search(r"gs://[^/]+/([^/]+)/", image_url)
        if not match:
            continue

        folder_name = match.group(1)
        for prediction in task.get("predictions", []):
            mv = prediction.get("model_version", "")
            if mv == "final_preannotation":
                prediction["model_version"] = f"final_preannotation_{folder_name}"

    out = output_path or input_path
    with open(out, "w") as f:
        json.dump(tasks, f, indent=2)

    print(f"Updated {len(tasks)} tasks -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update model_version with folder suffix")
    parser.add_argument("input", help="Path to task JSON file")
    parser.add_argument("-o", "--output", help="Output path (default: overwrite input)", default=None)
    args = parser.parse_args()
    update_model_versions(args.input, args.output)
