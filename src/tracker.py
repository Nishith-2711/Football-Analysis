import os
import pickle

import numpy as np
import pandas as pd
from ultralytics import YOLO

# Pickling raw ultralytics Results for hundreds of frames blows RAM; we persist
# only tensors/boxes as plain dicts (see detect_frames).
STUB_VERSION = 1


class _StubBoxes:
    def __init__(self, boxes):
        self._boxes = boxes

    def __len__(self):
        return len(self._boxes)

    def __iter__(self):
        return iter(self._boxes)


class _StubBox:
    """Minimal box view compatible with get_object_tracks."""

    def __init__(self, cls_id, xyxy, conf, track_id):
        self.cls = np.array([cls_id], dtype=np.float32)
        self.conf = np.array([conf], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.id = None if track_id is None else np.array(
            [float(track_id)], dtype=np.float32
        )


class _StubDetection:
    def __init__(self, names, boxes):
        self.names = names
        if boxes is None:
            self.boxes = None
        else:
            self.boxes = _StubBoxes(boxes)


def _result_to_frame_dict(result):
    """Strip a Results row down to plain Python / numpy-serializable data."""
    names = dict(result.names)
    if result.boxes is None or len(result.boxes) == 0:
        return names, []
    boxes = []
    for box in result.boxes:
        cls_id = int(box.cls.tolist()[0])
        xyxy = box.xyxy.tolist()[0]
        conf = float(box.conf[0]) if box.conf is not None else 1.0
        tid = None
        if box.id is not None:
            tid = int(box.id.tolist()[0])
        boxes.append({"cls": cls_id, "xyxy": xyxy, "conf": conf, "id": tid})
    return names, boxes


def _frame_dict_to_detection(names, box_dicts):
    boxes = [
        _StubBox(b["cls"], b["xyxy"], b["conf"], b["id"]) for b in box_dicts
    ]
    return _StubDetection(names, boxes)


class Tracker:
    """Object detection and tracking"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self, ball_tracks, max_gap=25):
        """Interpolate missing ball positions, but only across short gaps.

        Args:
            ball_tracks: dict  {1: {frame_num: {"bbox": [x1,y1,x2,y2]}, ...}}
            max_gap: int  Maximum number of consecutive missing frames to
                         interpolate across.  Gaps larger than this are left
                         as NaN (no ball) so that downstream code doesn't
                         use wildly inaccurate positions.
        """
        ball_dict = ball_tracks.get(1, {})
        if not ball_dict:
            return ball_tracks
        max_frame = max(ball_dict.keys())

        ball_positions = []
        for i in range(max_frame + 1):
            if i in ball_dict:
                ball_positions.append(ball_dict[i]["bbox"])
            else:
                ball_positions.append([None, None, None, None])

        df = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        # Only interpolate across gaps <= max_gap frames
        # Mark which rows were originally detected
        detected = df['x1'].notna()

        # Interpolate everything first, then null-out sections that were
        # part of a gap longer than max_gap
        df_interp = df.interpolate(limit=max_gap, limit_direction='forward')
        df_interp = df_interp.bfill(limit=max_gap)

        # Reconstruct – only include frames that have valid data
        new_ball_tracks = {1: {}}
        for i, row in enumerate(df_interp.to_numpy().tolist()):
            if not any(pd.isna(v) for v in row):
                new_ball_tracks[1][i] = {"bbox": row}

        return new_ball_tracks

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detect objects in frames"""
        if read_from_stub and stub_path is not None and os.path.exists(
                stub_path):
            print(f"Loading detections from {stub_path}")
            with open(stub_path, 'rb') as f:
                payload = pickle.load(f)
            return self._stub_payload_to_detections(payload)

        print(f"Detecting objects in {len(frames)} frames...")
        names_ref = None
        frame_boxes = []

        # We run BOTH track() and predict() on each frame:
        #   - track()  → players/referees get persistent IDs across frames
        #   - predict() → ball detections (the tracker's BoT-SORT filter
        #                  aggressively kills small/fast ball detections:
        #                  predict finds ball in ~90% of frames vs ~6% for track)
        # Ball boxes from predict() are merged into the frame's box list.

        ball_cls_id = None  # resolved on first frame

        for frame_num, frame in enumerate(frames):
            # --- 1. track() for players / referees with IDs ---
            track_results = self.model.track(frame, persist=True, conf=0.05,
                                              iou=0.5, verbose=False)
            r0 = track_results[0]
            names, boxes = _result_to_frame_dict(r0)
            if names_ref is None:
                names_ref = names

            # Resolve ball class id once
            if ball_cls_id is None:
                names_inv = {v: k for k, v in names_ref.items()}
                ball_cls_id = names_inv.get("ball")
                if ball_cls_id is None:
                    for cid, nm in names_ref.items():
                        if "ball" in (nm or "").lower():
                            ball_cls_id = int(cid)
                            break
                    if ball_cls_id is None:
                        ball_cls_id = 0

            # Check if track() already found a ball
            has_tracked_ball = any(b["cls"] == ball_cls_id for b in boxes)

            # --- 2. predict() to recover ball detections the tracker missed ---
            if not has_tracked_ball:
                pred_results = self.model.predict(frame, conf=0.05,
                                                   verbose=False)
                for box in pred_results[0].boxes:
                    cls_id = int(box.cls.tolist()[0])
                    if cls_id == ball_cls_id:
                        xyxy = box.xyxy.tolist()[0]
                        conf = float(box.conf[0]) if box.conf is not None else 1.0
                        boxes.append({
                            "cls": cls_id,
                            "xyxy": xyxy,
                            "conf": conf,
                            "id": None,  # ball doesn't need a track id
                        })
                        break  # only need the best ball detection
                del pred_results

            frame_boxes.append(boxes)

            del track_results, r0
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            if frame_num % 50 == 0:
                print(f"Processed {frame_num}/{len(frames)} frames")

        detections = [
            _frame_dict_to_detection(names_ref, b) for b in frame_boxes
        ]

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            payload = {
                "version": STUB_VERSION,
                "names": names_ref,
                "frames": frame_boxes,
            }
            with open(stub_path, 'wb') as f:
                pickle.dump(payload, f)
            print(f"Saved detections to {stub_path}")

        return detections

    def _stub_payload_to_detections(self, payload):
        """Load stub written by detect_frames or legacy full-Results list."""
        if isinstance(payload, dict) and payload.get("version") == STUB_VERSION:
            names = payload["names"]
            return [
                _frame_dict_to_detection(names, b)
                for b in payload["frames"]
            ]
        # Legacy: list of ultralytics Results — may OOM when pickling; still load if present
        if isinstance(payload, list) and payload:
            return payload
        raise ValueError("Unrecognized track stub format; delete the .pkl and re-run.")

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """Get tracks organized by object type"""
        detections = self.detect_frames(frames, read_from_stub, stub_path)

        tracks = {
            "players": {},   #tracks["players"] = { player_id_1: {frame_num_1: { "bbox": [x1, y1, x2, y2] },
            "referees": {},
            "ball": {}
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            # Check if any detections exist
            if detection.boxes is None or len(detection.boxes) == 0:
                continue

            def _cls_name(cid):
                n = cls_names.get(int(cid), "")
                return (n or "").lower()

            ball_cls_id = cls_names_inv.get("ball")
            if ball_cls_id is None:
                for cid, nm in cls_names.items():
                    if "ball" in (nm or "").lower():
                        ball_cls_id = int(cid)
                        break
                if ball_cls_id is None:
                    ball_cls_id = 2  # common default when names lack 'ball'

            # Ball: do not require box.id — the tracker often leaves id=None for a
            # small fast object, which was dropping most ball boxes.
            ball_candidates = []
            for box in detection.boxes:
                cls_id = int(box.cls.tolist()[0])
                if cls_id != ball_cls_id and _cls_name(cls_id) != "ball":
                    continue
                conf = float(box.conf[0]) if box.conf is not None else 1.0
                bbox = box.xyxy.tolist()[0]
                ball_candidates.append((conf, bbox))
            if ball_candidates:
                _, best_bbox = max(ball_candidates, key=lambda x: x[0])
                if 1 not in tracks["ball"]:
                    tracks["ball"][1] = {}
                tracks["ball"][1][frame_num] = {"bbox": best_bbox}

            # Players / referees: need a track id for identity across frames
            for box in detection.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.tolist()[0])
                bbox = box.xyxy.tolist()[0]
                cls_id = int(box.cls.tolist()[0])

                if cls_id == ball_cls_id or _cls_name(cls_id) == "ball":
                    continue

                # Determine object type (goalkeepers are tracked as players)
                gk_cls_id = cls_names_inv.get('goalkeeper', -1)
                if cls_id == cls_names_inv.get('player', 0) or cls_id == gk_cls_id:
                    if track_id not in tracks["players"]:
                        tracks["players"][track_id] = {}
                    tracks["players"][track_id][frame_num] = {
                        "bbox": bbox,
                        "is_goalkeeper": (cls_id == gk_cls_id),
                    }

                elif cls_id == cls_names_inv.get('referee', 1):
                    if track_id not in tracks["referees"]:
                        tracks["referees"][track_id] = {}
                    tracks["referees"][track_id][frame_num] = {
                        "bbox": bbox
                    }

        print(
            f"Tracked {len(tracks['players'])} players, {len(tracks['referees'])} referees")
        return tracks
