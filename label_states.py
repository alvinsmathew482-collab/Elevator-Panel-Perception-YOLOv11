import cv2
import os
import json
import glob

# ---- Paths ----
TEST_IMAGES = "datasets/final/test/images"
TEST_LABELS = "datasets/final/test/labels"
OUTPUT_FILE = "ground_truth_states.json"

CLASS_NAMES = ['1_floor', '2_floor', '3_floor', 'B_floor', 'G_floor',
               'bell_unpressed', 'door_close', 'door_open',
               'panel_region', 'telephone_unpressed']

# class not needed: panel_region
SKIP_CLASSES = {'panel_region'}

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

def main():
    
    results = []
    done_keys = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            results = json.load(f)
        done_keys = {(r["image"], r["box"]) for r in results}
        print(f"Resuming — {len(results)} already labeled")

    image_paths = sorted(glob.glob(os.path.join(TEST_IMAGES, "*.jpg")))
    if not image_paths:
        print(f"No images found in {TEST_IMAGES}")
        return

    print("\n=== State Labeling ===")
    print("For each button crop:")
    print("  p = pressed")
    print("  u = unpressed")
    print("  s = skip this button")
    print("  q = quit and save\n")

    cv2.namedWindow("Label State", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Label State", 500, 500)

    quit_flag = False

    for img_path in image_paths:
        if quit_flag:
            break

        img_name = os.path.basename(img_path)
        label_path = os.path.join(TEST_LABELS,
                                  img_name.replace(".jpg", ".txt"))

        if not os.path.exists(label_path):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path) as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for box_idx, line in enumerate(lines):
            if quit_flag:
                break

            
            if (img_name, box_idx) in done_keys:
                continue

            parts = line.split()
            cls_id = int(parts[0])
            cls_name = CLASS_NAMES[cls_id]

            if cls_name in SKIP_CLASSES:
                continue

            x1, y1, x2, y2 = polygon_to_pixels(parts, w, h)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            # Enlarge crop for visibility
            display = cv2.resize(crop, (400, 400),
                                 interpolation=cv2.INTER_NEAREST)

            # Add label banner
            banner = display.copy()
            cv2.rectangle(banner, (0, 0), (400, 40), (0, 0, 0), -1)
            cv2.putText(banner, f"{cls_name}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Label State", banner)

            # Wait for a valid keypress
            state = None
            while state is None:
                key = cv2.waitKey(0) & 0xFF
                if key == ord('p'):
                    state = "pressed"
                elif key == ord('u'):
                    state = "unpressed"
                elif key == ord('s'):
                    state = "skip"
                elif key == ord('q'):
                    quit_flag = True
                    break

            if quit_flag or state == "skip":
                continue

            results.append({
                "image": img_name,
                "box": box_idx,
                "class": cls_name,
                "true_state": state
            })

            # Save after every label
            with open(OUTPUT_FILE, "w") as f:
                json.dump(results, f, indent=2)

    cv2.destroyAllWindows()

    # Summary
    pressed = sum(1 for r in results if r["true_state"] == "pressed")
    unpressed = sum(1 for r in results if r["true_state"] == "unpressed")
    print(f"\nDone. Total labeled: {len(results)}")
    print(f"  Pressed:   {pressed}")
    print(f"  Unpressed: {unpressed}")
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
