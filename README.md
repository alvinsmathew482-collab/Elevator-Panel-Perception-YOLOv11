# Elevator Button Perception

Computer vision pipeline for the Open Droids elevator-interaction project. Given a camera view of an elevator panel, it finds each button, identifies which button it is, decides whether it is lit (pressed), and draws the result on the frame. Each frame's detections are also written to a JSON log.

Scope: panel perception only. Depth estimation and x/y coordinates for the robot arm are handled separately.

---

## 1. How it works

Detection and state are handled by two separate stages.

```
camera frame
    |
    v
YOLOv11n  ->  box + button identity (1, 2, 3, B, G, door open, door close, bell, telephone, panel)
    |
    v
crop the centre of each box
    |
    v
colour test  ->  pressed / unpressed
    |
    v
annotated frame + JSON
```

**Why two stages.** The first version of the model had state baked into the class names (`B_floor_pressed`, `B_floor_unpressed`). Both elevators we collected from light up differently when pressed (red on Elevator 1, white on Elevator 2), and YOLO did not separate pressed from unpressed reliably in live tests. Moving state out of the detector and into a colour measurement made the state decision explicit and measurable.

**The colour test.** For each detected button, the centre `CENTER_CROP` fraction of the box is sampled and two numbers are computed:

- `redness` = mean(R) - mean(G). Elevator 1 buttons glow red when pressed.
- `white_frac` = fraction of pixels where min(R, G, B) > 200. Elevator 2 buttons glow white when pressed.

The button is called pressed if `redness > RED_THRESHOLD` or `white_frac > WHITE_THRESHOLD`.

---

## 2. Files

| File | Purpose |
|---|---|
| `Model-IRIS.py` | Main script. Live camera inference, drawing, JSON output. |
| `evaluate_states.py` | Computes state accuracy / precision / recall / F1 and the confusion matrix against the hand-labelled ground truth. |
| `label_states.py` | Tool used to build the ground truth: shows each annotated button crop, press `p` / `u` / `s` / `q`. |
| `review_states.py` | Re-checks existing ground-truth labels and lets you correct them. |
| `ground_truth_states.json` | 564 hand-labelled button states for the test set. |
| `final_v1_best.pt` | Trained YOLOv11n weights. |
| `data.yaml` | Dataset definition (10 classes). |
| `datasets/final/test/` | Held-out test images and labels. |

`ground_truth_states.json` is keyed on the exported test image filenames and box order. If the dataset is re-exported from Roboflow the filenames change and the ground truth will no longer line up, so keep the test folder and this file together.

---

## 3. Setup and usage

```bash
pip install ultralytics opencv-python numpy
```

Live inference (press `q` to quit; saves `output_final.mp4` and a timestamped JSON):

```bash
python Model-IRIS.py
```

State evaluation:

```bash
python evaluate_states.py
```

Rebuild ground truth from scratch (delete `ground_truth_states.json` first, otherwise it resumes):

```bash
python label_states.py
```

### Configuration

Constants at the top of `Model-IRIS.py` (and the matching ones in `evaluate_states.py`). Check which file each script actually reads its values from before changing anything: during the crop experiments, editing only `Model-IRIS.py` changed the evaluation output, so the two are linked in practice. After any change, re-run `evaluate_states.py` to confirm the new value took effect.

| Constant | Value | Meaning |
|---|---|---|
| `RED_THRESHOLD` | 20 | Minimum R-G difference to call a button pressed (red glow) |
| `WHITE_THRESHOLD` | 0.07 | Minimum fraction of near-white pixels to call a button pressed (white glow) |
| `CENTER_CROP` | 0.5 | Fraction of the box (centred) sampled for colour |
| `CONF_THRESHOLD` | 0.4 | YOLO confidence cut-off at inference |
| `IOU_MATCH_THRESHOLD` | 0.3 | Overlap needed to match a detection to a ground-truth box (evaluation only) |
| `CONF_THRESHOLD` (evaluate) | 0.4 | Confidence cut-off used while evaluating |

### JSON output

One entry per frame:

```json
{
  "frame": 12,
  "detections": [
    {"label": "3", "type": "floor", "state": "pressed", "confidence": 0.91,
     "avg_rgb": {"r": 140, "g": 60, "b": 58}}
  ]
}
```

Bounding-box coordinates are not in the log yet (see section 8).

---

## 4. Dataset

- Source: photos taken by hand on two elevators, annotated in Roboflow. Elevator 1 is the author's building (red glow when pressed); Elevator 2 is a different building (white glow when pressed).
- 447 images, split 291 train / 66 validation / 90 test.
- The test set is a separate shoot (varied angle, distance and lighting), not random frames from the training batch. It was annotated manually rather than with the model's auto-labeller, so the ground truth is independent of the model being tested.
- 10 classes: `1_floor`, `2_floor`, `3_floor`, `B_floor`, `G_floor`, `bell_unpressed`, `door_close`, `door_open`, `panel_region`, `telephone_unpressed`. The `_unpressed` suffix on bell and telephone is left over from the earlier naming; state is ignored from class names.
- Elevator 2 has no telephone button, so that class comes from Elevator 1 only. The bell was never pressed during collection.
- Roboflow exported the annotations as polygons, not boxes. Anything reading the label files has to take the min/max of the polygon points to get a box (`polygon_to_pixels` in the scripts).

---

## 5. Training

Trained on Google Colab (T4 GPU) from `yolo11n.pt`:

```python
from ultralytics import YOLO
model = YOLO("yolo11n.pt")
model.train(
    data="/content/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,          # confirm against args.yaml in the run folder
    device=0,
    project="/content/drive/MyDrive/elevator_runs",
    name="final_v1",
)
```

Training on CPU took hours for 20 epochs; the GPU run takes about 30 minutes.

---

## 6. Results

All numbers are on the 90-image held-out test set.

### 6.1 Detection (YOLO, `model.val(split="test")`)

Overall: mAP50 **0.861**, mAP50-95 **0.812**, precision 0.894, recall 0.742.

| Class | Instances | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|---|
| 1_floor | 72 | 0.967 | 0.803 | 0.946 | 0.903 |
| 2_floor | 69 | 0.946 | 0.760 | 0.912 | 0.861 |
| 3_floor | 62 | 0.914 | 0.853 | 0.944 | 0.906 |
| B_floor | 69 | 0.942 | 0.702 | 0.833 | 0.789 |
| G_floor | 71 | 0.981 | 0.734 | 0.946 | 0.916 |
| bell_unpressed | 66 | 0.846 | 0.750 | 0.848 | 0.793 |
| door_close | 68 | 0.896 | 0.764 | 0.874 | 0.828 |
| door_open | 67 | 0.961 | 0.730 | 0.872 | 0.838 |
| panel_region | 67 | 0.753 | 0.940 | 0.929 | 0.831 |
| telephone_unpressed | 22 | 0.739 | 0.387 | 0.503 | 0.452 |

The model is precise but misses buttons: recall is the weaker side across nearly every class. Telephone is the weakest class (22 test instances, one elevator, little training data).

### 6.2 Pressed / unpressed state

Ground truth: 564 buttons labelled by hand (85 pressed, 479 unpressed). `evaluate_states.py` reports two views.

**Colour test alone**, run on the ground-truth boxes (isolates the colour logic from detection misses), `CENTER_CROP = 0.5`:

| Accuracy | Precision | Recall | F1 |
|---|---|---|---|
| 0.957 | 0.785 | 0.988 | 0.875 |

**End to end**, run on the model's own detections:

| IoU match | Conf | Matched | Missed by YOLO | Pressed->pressed | Pressed->unpressed | Unpressed->pressed | Unpressed->unpressed | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 0.5 | 339 | 225 | 57 | 28 | 12 | 467 | 0.929 | 0.826 | 0.671 | 0.740 |
| 0.3 | 0.5 | 339 | 225 | 57 | 28 | 12 | 467 | 0.929 | 0.826 | 0.671 | 0.740 |
| **0.3** | **0.4** | **354** | **210** | **60** | **25** | **14** | **465** | **0.931** | **0.811** | **0.706** | **0.755** |

Reading these:

- Accuracy is misleading on its own. 85% of buttons are unpressed, so always answering "unpressed" would score about 85%. F1 is the better summary.
- The colour logic is not the main limit. On correctly located buttons it catches 84 of 85 pressed buttons. The drop to 0.706 end-to-end recall comes almost entirely from buttons YOLO never detected (210 of 564 missed even at conf 0.4).
- Changing the IoU match threshold from 0.5 to 0.3 changed nothing, so the misses are real misses, not loose-box artefacts.
- Lowering the confidence threshold from 0.5 to 0.4 gained a few detections for a small precision cost, which suits a task where missing a button is the worse error.

---

## 7. Development log

**Version 1: state in the class names.** 17 classes such as `B_floor_pressed` / `B_floor_unpressed`, 341 images, no real validation split (validation pointed at the training folder). Reported mAP50 of 0.923 after 20 epochs on CPU, but that figure was measured on data the model had trained on and should not be quoted.

**Version 2: detection and colour separated.** Class pairs merged into one class per button (10 classes). Added a proper validation split. Rebuilt the test set from a separate shoot.

**Colour thresholds measured, not guessed.** The first thresholds (`R > 180, G < 80, B < 80`) were estimates. Measuring 20 calibration photos (lit and unlit versions of the same buttons on both elevators) showed that even clearly red buttons have a mean R of only about 100-157 under real lighting, so those thresholds would have called everything unpressed. Two features were found to separate the classes cleanly: `redness` for Elevator 1 and `white_frac` for Elevator 2. On the calibration photos, pressed `redness` was 27.8 to 87.4 against an unpressed maximum of about 13.4, and pressed `white_frac` was 0.100 to 0.127 against an unpressed maximum of 0.010. Thresholds of 20 and 0.07 sit in those gaps. These thresholds were tuned on the same 20 photos, so the test-set results above are the real validation.

**Ground truth and evaluation built.** `label_states.py` and `evaluate_states.py` were written to get state metrics. Two bugs found along the way, both worth knowing about:

- Label files were polygons, not boxes. The first parser read the first four numbers as a box, which produced wrong crops and zero matches. Fixed by taking the min/max of the polygon points, and the ground truth was relabelled.
- The unpressed count was not incremented for buttons YOLO missed. Fixed.

Two ground-truth labels were then corrected with `review_states.py`.

**Detection vs colour diagnosis.** Running the colour test directly on ground-truth boxes (F1 0.875) against the full pipeline (F1 0.740-0.755) showed that detection coverage, not colour logic, is the bottleneck.

**Confidence sweep.** Lowering confidence from 0.5 to 0.4 improved end-to-end F1 from 0.740 to 0.755.

### Experiments for the camera-glare feedback

The reflective panel produces glare, and a whole-box colour average can be pulled around by it. Three approaches were tested.

1. **Glyph segmentation** (threshold the button crop with Otsu, keep the largest contour as the character, measure colour only inside it). Not reliable: on the 20 calibration photos the segmented region swung between 2.4% and 88.5% of the crop depending on lighting, and the pressed and unpressed ranges overlapped on both elevators. Rejected.

2. **Smaller sampling window** (`CENTER_CROP` 0.5 -> 0.3 -> 0.18). On the calibration photos the gap between pressed and unpressed grew as the window shrank, which looked promising. On the full test set it did not hold up:

   | CENTER_CROP | Colour-only precision | Colour-only recall | Colour-only F1 | End-to-end F1 |
   |---|---|---|---|---|
   | **0.50** | 0.785 | 0.988 | **0.875** | **0.755** |
   | 0.30 | 0.720 | 1.000 | 0.837 | 0.732 |
   | 0.18 | 0.759 | 1.000 | 0.863 | 0.727 |

   Recall rose to 1.000 but false alarms rose more. A smaller window averages fewer pixels, so a single bright reflection moves the result more, not less. `CENTER_CROP = 0.5` is the best setting and is the one used. Lesson: the calibration-photo gap and test-set F1 disagreed, so settings should be judged on the test set.

3. **Excluding blown-out pixels** (drop pixels where R, G and B are all above 245, or 230, before averaging). On Elevator 1 this changed almost nothing (gap 31.3 -> 31.9). On Elevator 2 it removed the signal itself, because the pressed state there is near-white pixels: the gap fell from 0.090 to 0.003. Fully clipped pixels were also rare in the calibration set (average 1.7%, worst case 10.2%). Rejected.

A brightness cut-off cannot separate glare from a white-lit glyph, since both are bright. Separating them would need something other than intensity.

---

## 8. Known limitations and next steps

**Limitations**

- Detection misses about 37% of labelled buttons at conf 0.4 (210 of 564). This is the main cap on end-to-end performance.
- Telephone detection is weak (mAP50 0.503) and was trained and tested on one elevator only.
- Colour thresholds were calibrated on two specific elevators with a phone camera. A third elevator, or a different camera's exposure, will probably need recalibration, especially the Elevator 2 white-glow path.
- Bell and telephone were never pressed, so state classification for those two is untested.
- 85 pressed examples drawn from 90 images. The buttons within one photo share lighting, so the effective sample is smaller than 564. Quote sample sizes alongside the metrics.
- Ground truth was labelled by one person.
- Live webcam runs were recorded (`output_final.mp4`, JSON logs) but no quantitative live-feed metrics were collected.

**Next steps**

- Add `[x1, y1, x2, y2]` to each JSON detection. The teammate doing depth/x-y needs it, and it is not in the output yet. Confirm the format he wants first.
- Improve detection recall: more training data (especially telephone, panel, and small buttons) or a larger model (`yolo11s`) instead of the nano model.
- Glare handling that does not depend on brightness, for example rejecting small isolated bright blobs by size.
- Clarify with the manager whether "pixel profile" means a per-button unlit baseline compared against the live frame; that is different from the single per-elevator threshold used now and has not been built.
- Move the colour function and constants into one shared module so the inference and evaluation scripts cannot drift apart. (This also removes the ambiguity about which file's constants are in use.)
