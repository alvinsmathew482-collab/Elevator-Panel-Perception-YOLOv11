import cv2, json, os

GROUND_TRUTH_FILE = "ground_truth_states.json"
TEST_IMAGES = "datasets/final/test/images"
TEST_LABELS = "datasets/final/test/labels"

CLASS_NAMES = ['1_floor', '2_floor', '3_floor', 'B_floor', 'G_floor',
               'bell_unpressed', 'door_close', 'door_open',
               'panel_region', 'telephone_unpressed']

ONLY_REVIEW = "pressed"   # set to None to review everything

def polygon_to_pixels(parts, img_w, img_h):
    coords = list(map(float, parts[1:]))
    xs, ys = coords[0::2], coords[1::2]
    return (int(min(xs)*img_w), int(min(ys)*img_h),
            int(max(xs)*img_w), int(max(ys)*img_h))

with open(GROUND_TRUTH_FILE) as f:
    records = json.load(f)

cv2.namedWindow("Review", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Review", 500, 500)

print("p = mark pressed | u = mark unpressed | n = keep as is | q = quit and save\n")

for rec in records:
    if ONLY_REVIEW and rec["true_state"] != ONLY_REVIEW:
        continue

    img_path = os.path.join(TEST_IMAGES, rec["image"])
    label_path = os.path.join(TEST_LABELS, rec["image"].replace(".jpg", ".txt"))
    if not os.path.exists(img_path):
        continue

    img = cv2.imread(img_path)
    h, w = img.shape[:2]
    with open(label_path) as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    parts = lines[rec["box"]].split()
    x1, y1, x2, y2 = polygon_to_pixels(parts, w, h)
    crop = img[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
    if crop.size == 0:
        continue

    disp = cv2.resize(crop, (400, 400), interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(disp, (0,0), (400,40), (0,0,0), -1)
    cv2.putText(disp, f"{rec['class']} = {rec['true_state']}", (10,28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
    cv2.imshow("Review", disp)

    while True:
        k = cv2.waitKey(0) & 0xFF
        if k == ord('p'):
            rec["true_state"] = "pressed"; break
        if k == ord('u'):
            rec["true_state"] = "unpressed"; break
        if k == ord('n'):
            break
        if k == ord('q'):
            cv2.destroyAllWindows()
            with open(GROUND_TRUTH_FILE, "w") as f:
                json.dump(records, f, indent=2)
            print("Saved.")
            raise SystemExit

    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(records, f, indent=2)

cv2.destroyAllWindows()
p = sum(1 for r in records if r["true_state"] == "pressed")
u = sum(1 for r in records if r["true_state"] == "unpressed")
print(f"Done. Pressed: {p}  Unpressed: {u}")