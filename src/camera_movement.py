import os
import pickle

import cv2
import numpy as np


class CameraMovementEstimator:
    """Estimate camera movement using optical flow"""

    def __init__(self, frame):
        self.minimum_distance = 5

        # Features for tracking
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.features = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask_features
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """Calculate camera movement for all frames"""
        if read_from_stub and stub_path is not None and os.path.exists(
                stub_path):
            print(f"Loading camera movement from {stub_path}")
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        print("Calculating camera movement...")
        camera_movement = [[0, 0]] * len(frames)

        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        for frame_num in range(1, len(frames)):
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)

            if old_features is None or len(old_features) == 0:
                old_features = cv2.goodFeaturesToTrack(frame_gray,
                                                       **self.features)
                continue

            new_features, status, error = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            if new_features is None:
                camera_movement[frame_num] = [0, 0]
                continue

            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            for i, (new, old) in enumerate(zip(new_features, old_features)):
                if status[i]:
                    new_features_point = new.ravel()
                    old_features_point = old.ravel()

                    distance = np.linalg.norm(
                        new_features_point - old_features_point)
                    if distance > max_distance:
                        max_distance = distance
                        camera_movement_x, camera_movement_y = new_features_point - old_features_point

            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x,
                                              camera_movement_y]
                old_features = cv2.goodFeaturesToTrack(frame_gray,
                                                       **self.features)

            old_gray = frame_gray.copy()

            if frame_num % 50 == 0:
                print(
                    f"Processed camera movement for {frame_num}/{len(frames)} frames")

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)
            print(f"Saved camera movement to {stub_path}")

        return camera_movement

    def adjust_positions_to_camera_movement(self, tracks, camera_movement):
        """Adjust object positions based on camera movement"""
        for object_type, object_tracks in tracks.items():
            for object_id, object_track in object_tracks.items():
                for frame_num, track_info in object_track.items():
                    if 'position' not in track_info:
                        continue

                    position = track_info['position']
                    camera_adjustment = camera_movement[frame_num]

                    position_adjusted = (
                        position[0] - camera_adjustment[0],
                        position[1] - camera_adjustment[1]
                    )
                    tracks[object_type][object_id][frame_num][
                        'position_adjusted'] = position_adjusted
