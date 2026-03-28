import numpy as np


class SpeedAndDistanceEstimator:
    """Calculate player speed and distance"""

    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self, tracks):
        """Add speed and distance metrics to tracks"""
        total_distance = {}

        for object_type, object_tracks in tracks.items():
            if object_type == "ball":
                continue

            for object_id, object_track in object_tracks.items():
                if object_id not in total_distance:
                    total_distance[object_id] = 0

                for frame_num, track_info in sorted(object_track.items()):
                    if frame_num == 0:
                        track_info['speed'] = 0
                        track_info['distance'] = 0
                        continue

                    last_frame = max(
                        [f for f in object_track.keys() if f < frame_num],
                        default=None)

                    if (last_frame is not None and
                            'position_transformed' in track_info and
                            'position_transformed' in object_track[last_frame]):

                        position_transformed = track_info[
                            'position_transformed']
                        last_position_transformed = object_track[last_frame][
                            'position_transformed']

                        distance_covered = np.linalg.norm(
                            position_transformed - last_position_transformed)

                        time_elapsed = (
                                                   frame_num - last_frame) / self.frame_rate
                        speed_meters_per_second = distance_covered / time_elapsed
                        speed_km_per_hour = speed_meters_per_second * 3.6

                        total_distance[object_id] += distance_covered

                        track_info['speed'] = speed_km_per_hour
                        track_info['distance'] = total_distance[object_id]
                    else:
                        track_info['speed'] = 0
                        track_info['distance'] = total_distance.get(object_id,
                                                                    0)
