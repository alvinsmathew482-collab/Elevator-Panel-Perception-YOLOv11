"""
Evaluates the colour-based pressed/unpressed classifier against
manually labeled ground truth, on the model's own detections.

Run from the project root:
    python evaluate_states.py
"""

import cv2
import json
import os
import numpy as np
from ultralytics import YOLO

# ---- Paths — update MODEL_PATH to your downloaded best.pt ----
MODEL_PATH = "final_v1_best.pt"
TEST_IMAGES = "datasets/final/test/images"
TEST_LABELS = "datasets/final/test/labels"
GROUND_TRUTH_FILE = "ground_truth_states.json"

CLASS_NAMES = ['1_floor', '2_floor', '3_floor', 'B_floor', 'G_floor',
               'bell_unpressed', 'door_close', 'door_open',
               'panel_region', 'telephone_unpressed']

SKIP_CLASSES = {'panel_region'}

CONF_THRESHOLD = 0.4
IOU_MATCH_THRESHOLD = 0.3   # how much overlap counts as "same button"

# ---- Colour thresholds, calibrated from measured samples ----
RED_THRESHOLD = 20
WHITE_THRESHOLD = 0.07
CENTER_CROP = 0.5


def get_button_state(frame, x1, y1, x2, y2):
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return "unpressed"
    dx, dy = int(w * (1 - CENTER_CROP) / 2), int(h * (1 - CENTER_CROP) / 2)
    crop = frame[y1 + dy:y2 - dy, x1 + dx:x2 - dx]
    if crop.size == 0:
        return "unpressed"
    px = crop.reshape(-1, 3).astype(float)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]
    redness = r.mean() - g.mean()
    min_ch = np.minimum(np.minimum(r, g), b)
    white_frac = float((min_ch > 200).mean())
    if redness > RED_THRESHOLD:
        return "pressed"
    if white_frac > WHITE_THRESHOLD:
        return "pressed"
    return "unpressed"


def polygon_to_pixels(parts, img_w, img_h):
    """
    Roboflow exported these annotations as polygons, not boxes.
    Line format: class x1 y1 x2 y2 x3 y3 ...  (normalized)
    We take the min/max of the points to get the enclosing bounding box.
    """
    coords = list(map(float, parts[1:]))
    xs = coords[0::2]
    ys = coords[1::2]
    x1 = int(min(xs) * img_w)
    y1 = int(min(ys) * img_h)
    x2 = int(max(xs) * img_w)
    y2 = int(max(ys) * img_h)
    return x1, y1, x2, y2

def iou(boxA, boxB):
    xa1, ya1, xa2, ya2 = boxA
    xb1, yb1, xb2, yb2 = boxB
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    areaA = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    areaB = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = areaA + areaB - inter
    return inter / union if union > 0 else 0

def main():
    if not os.path.exists(GROUND_TRUTH_FILE):
        print(f"Missing {GROUND_TRUTH_FILE} — run label_states.py first")
        return
    if not os.path.exists(MODEL_PATH):
        print(f"Missing {MODEL_PATH} — update MODEL_PATH to your best.pt location")
        return

    with open(GROUND_TRUTH_FILE) as f:
        gt_records = json.load(f)

    tp_pressed = 0
    fp_pressed = 0
    fn_pressed = 0
    tn_unpressed = 0

    c_tp = c_fp = c_fn = c_tn = 0   # colour-only, on ground-truth boxes
# Group ground truth by image, rebuild pixel boxes from the label files
    gt_by_image = {}
    for rec in gt_records:
        gt_by_image.setdefault(rec["image"], []).append(rec)

    model = YOLO(MODEL_PATH)

    # confusion matrix counts
    tp_pressed = 0   # predicted pressed, true pressed
    fp_pressed = 0   # predicted pressed, true unpressed    
    fn_pressed = 0   # predicted unpressed, true pressed  (also covers missed detections)
    tn_unpressed = 0 # predicted unpressed, true unpressed

    matched_count = 0
    unmatched_count = 0
    mismatched_class_count = 0

    for img_name, gt_list in gt_by_image.items():
        img_path = os.path.join(TEST_IMAGES, img_name)
        label_path = os.path.join(TEST_LABELS, img_name.replace(".jpg", ".txt"))
        if not os.path.exists(img_path) or not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Rebuild the ground-truth pixel boxes (same indexing as label_states.py)
        with open(label_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        gt_boxes = {}  # box_idx -> (x1,y1,x2,y2,class_name)
        for box_idx, line in enumerate(lines):
            parts = line.split()
            cls_id = int(parts[0])
            cls_name = CLASS_NAMES[cls_id]

            if cls_name in SKIP_CLASSES:
                continue    

            x1, y1, x2, y2 = polygon_to_pixels(parts, w, h)
            gt_boxes[box_idx] = (x1, y1, x2, y2, cls_name)

        # Run the model on this image
        results = model(img, conf=CONF_THRESHOLD, verbose=False)
        pred_boxes = []
        for box in results[0].boxes:
            cls_id = int(box.cls)
            cls_name = model.names[cls_id]
            if cls_name in SKIP_CLASSES:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pred_boxes.append((x1, y1, x2, y2, cls_name))

        # For each ground-truth labeled button, find the best-matching prediction
        for rec in gt_list:
            box_idx = rec["box"]
            true_state = rec["true_state"]
            if box_idx not in gt_boxes:
                continue
            gx1, gy1, gx2, gy2, gt_cls = gt_boxes[box_idx]

            # Colour classifier on ground-truth boxes, isolated from detection failures
            colour_only = get_button_state(img, gx1, gy1, gx2, gy2)
            if true_state == "pressed" and colour_only == "pressed":
                c_tp += 1
            elif true_state == "unpressed" and colour_only == "pressed":
                c_fp += 1
            elif true_state == "pressed" and colour_only == "unpressed":
                c_fn += 1
            else:
                c_tn += 1

            best_iou = 0
            best_pred = None
            for px1, py1, px2, py2, pcls in pred_boxes:
                score = iou((gx1, gy1, gx2, gy2), (px1, py1, px2, py2))
                if score > best_iou:
                    best_iou = score
                    best_pred = (px1, py1, px2, py2, pcls)

            if best_pred is None or best_iou < IOU_MATCH_THRESHOLD:
                # Model missed this button entirely.
                # It can't be state-scored, but a missed pressed button
                # is a real-world failure -> count as false negative.
                unmatched_count += 1
                if true_state == "pressed":
                    fn_pressed += 1
                else:
                    tn_unpressed += 1   # missed, but correctly not reported as pressed
                continue

            px1, py1, px2, py2, pcls = best_pred
            if pcls != gt_cls:
                mismatched_class_count += 1

            pred_state = get_button_state(img, px1, py1, px2, py2)
            matched_count += 1

            if true_state == "pressed" and pred_state == "pressed":
                tp_pressed += 1
            elif true_state == "unpressed" and pred_state == "pressed":
                fp_pressed += 1
            elif true_state == "pressed" and pred_state == "unpressed":
                fn_pressed += 1
            elif true_state == "unpressed" and pred_state == "unpressed":
                tn_unpressed += 1

    total = tp_pressed + fp_pressed + fn_pressed + tn_unpressed
    accuracy = (tp_pressed + tn_unpressed) / total if total else 0
    precision = tp_pressed / (tp_pressed + fp_pressed) if (tp_pressed + fp_pressed) else 0
    recall = tp_pressed / (tp_pressed + fn_pressed) if (tp_pressed + fn_pressed) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    c_total = c_tp + c_fp + c_fn + c_tn
    c_acc = (c_tp + c_tn) / c_total if c_total else 0
    c_prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) else 0
    c_rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) else 0
    c_f1 = (2 * c_prec * c_rec / (c_prec + c_rec)) if (c_prec + c_rec) else 0

    print("\n=== Colour classifier alone (on ground-truth boxes) ===")
    print(f"Accuracy:  {c_acc:.3f}")
    print(f"Precision: {c_prec:.3f}")
    print(f"Recall:    {c_rec:.3f}")
    print(f"F1 score:  {c_f1:.3f}")

    print("\n=== Detection coverage ===")
    print(f"Ground-truth labeled buttons: {len(gt_records)}")
    print(f"Matched to a model detection: {matched_count}")
    print(f"Missed by the model entirely: {unmatched_count}")
    print(f"Matched but wrong button class: {mismatched_class_count}")

    print("\n=== Confusion matrix (state, on matched detections + missed-pressed) ===")
    print(f"{'':15}{'Pred: Pressed':>15}{'Pred: Unpressed':>17}")
    print(f"{'True: Pressed':15}{tp_pressed:>15}{fn_pressed:>17}")
    print(f"{'True: Unpressed':15}{fp_pressed:>15}{tn_unpressed:>17}")

    print("\n=== State classification metrics ===")
    print(f"Accuracy:  {accuracy:.3f}")
    print(f"Precision: {precision:.3f}  (of predicted-pressed, how many were really pressed)")
    print(f"Recall:    {recall:.3f}  (of truly pressed, how many were caught)")
    print(f"F1 score:  {f1:.3f}")
    print(f"\nSample size: {tp_pressed + fn_pressed} pressed, {fp_pressed + tn_unpressed} unpressed")


if __name__ == "__main__":
    main()