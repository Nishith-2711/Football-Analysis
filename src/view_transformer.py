import cv2
import numpy as np


class ViewTransformer:
    """Transform view from camera perspective to bird's eye view"""

    def __init__(self):
        court_width = 68
        court_length = 23.32

        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ])

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices, self.target_vertices
        )

    def transform_point(self, point):
        """Transform a single point"""
        p = (int(point[0]), int(point[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
        if not is_inside:
            return None

        reshaped_point = np.array([point], dtype=np.float32).reshape(-1, 1, 2)
        transformed_point = cv2.perspectiveTransform(reshaped_point,
                                                     self.perspective_transformer)
        return transformed_point.reshape(-1, 2)[0]

    def add_transformed_position_to_tracks(self, tracks):
        """Add transformed positions to tracks"""
        for object_type, object_tracks in tracks.items():
            for object_id, object_track in object_tracks.items():
                for frame_num, track_info in object_track.items():
                    if 'position_adjusted' not in track_info:
                        continue

                    position = track_info['position_adjusted']
                    transformed_position = self.transform_point(position)
                    if transformed_position is not None:
                        tracks[object_type][object_id][frame_num][
                            'position_transformed'] = transformed_position
