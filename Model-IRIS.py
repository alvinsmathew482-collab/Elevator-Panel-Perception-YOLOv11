from ultralytics import YOLO
import cv2
import json
import numpy as np
from datetime import datetime

MODEL_PATH = "final_v1_best.pt" #loading trained model
model = YOLO(MODEL_PATH)

# ---- Class identity map: internal name -> (label, type) ----
CLASS_META = {
    "1_floor": ("1", "floor"),
    "2_floor": ("2", "floor"),
    "3_floor": ("3", "floor"),
    "B_floor": ("B", "floor"),
    "G_floor": ("G", "floor"),
    "door_open": ("Door Open", "function"),
    "door_close": ("Door Close", "function"),
    "bell_unpressed": ("Bell", "function"),
    "telephone_unpressed": ("Telephone", "function"),
    "panel_region": ("Panel", "panel"),
}

# ---- Colour thresholds ----
RED_THRESHOLD = 20       # R - G difference -> Elevator 1 red glow
WHITE_THRESHOLD = 0.07   # fraction of near-white pixels -> Elevator 2 white glow
CENTER_CROP = 0.5      # sample 50% of the box, avoids the dark bezel
CONF_THRESHOLD = 0.4    


def parse_label(raw):
    """Get button identity from YOLO's class name. State is NOT read from here."""
    key = raw.replace("_pressed", "").replace("_unpressed", "") if raw != "panel_region" else raw
    label, btn_type = CLASS_META.get(raw, CLASS_META.get(key, (raw, "unknown")))
    return label, btn_type


def get_button_state(frame, x1, y1, x2, y2):
    """
    Determines pressed/unpressed purely from measured colour signatures.
    Elevator 1 buttons glow red when pressed  -> high (R - G)
    Elevator 2 buttons glow white when pressed -> many near-saturated pixels
    """
    h, w = y2 - y1, x2 - x1
    if h <= 0 or w <= 0:
        return "unpressed", (0, 0, 0)

    dx, dy = int(w * (1 - CENTER_CROP) / 2), int(h * (1 - CENTER_CROP) / 2)
    crop = frame[y1 + dy:y2 - dy, x1 + dx:x2 - dx]

    if crop.size == 0:
        return "unpressed", (0, 0, 0)

    px = crop.reshape(-1, 3).astype(float)
    b, g, r = px[:, 0], px[:, 1], px[:, 2]

    redness = r.mean() - g.mean()
    min_ch = np.minimum(np.minimum(r, g), b)
    white_frac = float((min_ch > 200).mean())

    avg_rgb = (r.mean(), g.mean(), b.mean())

    if redness > RED_THRESHOLD:
        return "pressed", avg_rgb
    if white_frac > WHITE_THRESHOLD:
        return "pressed", avg_rgb
    return "unpressed", avg_rgb


def draw_detection(frame, x1, y1, x2, y2, label, btn_type, state, conf):
    if btn_type == "panel":
        color = (255, 165, 0)
        display = f"Panel | {round(conf * 100)}%"
    elif state == "pressed":
        color = (0, 0, 255)
        display = f"{label} | {btn_type.capitalize()} | Pressed | {round(conf * 100)}%"
    else:
        color = (0, 255, 0)
        display = f"{label} | {btn_type.capitalize()} | Unpressed | {round(conf * 100)}%"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(display, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
    cv2.putText(frame, display, (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("output_final.mp4", fourcc, 30.0, (width, height))

    cv2.namedWindow("Elevator Perception - Final", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Elevator Perception - Final", 1280, 720)

    output_data = []
    frame_count = 0

    print("Running — press Q to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Camera not found or disconnected")
            break

        frame_count += 1
        results = model(frame, conf=CONF_THRESHOLD, verbose=False)
        frame_json = []

        for box in results[0].boxes:
            conf = float(box.conf)
            raw = model.names[int(box.cls)]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            label, btn_type = parse_label(raw)

            if btn_type == "panel":
                draw_detection(frame, x1, y1, x2, y2, label, btn_type, None, conf)
                continue

            state, avg_rgb = get_button_state(frame, x1, y1, x2, y2)
            draw_detection(frame, x1, y1, x2, y2, label, btn_type, state, conf)

            frame_json.append({
                "label": label,
                "type": btn_type,
                "state": state,
                "confidence": round(conf, 3),
                "avg_rgb": {
                    "r": round(avg_rgb[0]), "g": round(avg_rgb[1]), "b": round(avg_rgb[2])
                }
            })

        output_data.append({"frame": frame_count, "detections": frame_json})
        out.write(frame)
        cv2.imshow("Elevator Perception - Final", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    filename = f"output_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved video: output_final.mp4")
    print(f"Saved JSON: {filename}")


if __name__ == "__main__":
    main()
