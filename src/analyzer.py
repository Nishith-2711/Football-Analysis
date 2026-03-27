import cv2
import numpy as np
import pickle
from ultralytics import YOLO
import os
import pandas as pd


class VideoUtils:
    """Video reading and writing utilities"""

    @staticmethod
    def read_video(video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        print(f"Read {len(frames)} frames from {video_path}")
        return frames

    @staticmethod
    def save_video(frames, output_path, fps=24):
        if len(frames) == 0:
            print("No frames to save!")
            return
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        height, width = frames[0].shape[:2]
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        print(f"Saved {len(frames)} frames to {output_path}")


class Tracker:
    """Object detection and tracking"""

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def interpolate_ball_position(self, ball_position):
        ball_position = [x.get(1,{}).get("bbox", []) for x in ball_position]
        df_ball_position = pd.DataFrame(ball_position, columns=['x1','y1',
                                                                'x2', 'y2'])

        #interpolate missing values
        df_ball_position = df_ball_position.interpolate()
        df_ball_position = df_ball_position.bfill()

        ball_position = [{1:{"bbox":x}} for x in df_ball_position.to_numpy().tolist()]

        return ball_position

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """Detect objects in frames"""
        if read_from_stub and stub_path is not None and os.path.exists(
                stub_path):
            print(f"Loading detections from {stub_path}")
            with open(stub_path, 'rb') as f:
                detections = pickle.load(f)
            return detections

        print(f"Detecting objects in {len(frames)} frames...")
        detections = []

        # Process frames one by one for tracking
        for frame_num, frame in enumerate(frames):
            # Track with persistence to maintain IDs across frames
            results = self.model.track(frame, persist=True, conf=0.1, iou=0.5,
                                       verbose=False)
            detections.append(results[0])  # Get first (only) result

            if frame_num % 50 == 0:
                print(f"Processed {frame_num}/{len(frames)} frames")

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(detections, f)
            print(f"Saved detections to {stub_path}")

        return detections

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

            # Get detection info
            for box in detection.boxes:
                if box.id is None:
                    continue

                track_id = int(box.id.tolist()[0])
                bbox = box.xyxy.tolist()[0]
                cls_id = int(box.cls.tolist()[0])

                # Determine object type
                if cls_id == cls_names_inv.get('player', 0):
                    if track_id not in tracks["players"]:
                        tracks["players"][track_id] = {}
                    tracks["players"][track_id][frame_num] = {
                        "bbox": bbox
                    }

                elif cls_id == cls_names_inv.get('referee', 1):
                    if track_id not in tracks["referees"]:
                        tracks["referees"][track_id] = {}
                    tracks["referees"][track_id][frame_num] = {
                        "bbox": bbox
                    }

                elif cls_id == cls_names_inv.get('ball', 2):
                    if 1 not in tracks["ball"]:
                        tracks["ball"][1] = {}
                    tracks["ball"][1][frame_num] = {
                        "bbox": bbox
                    }

        print(
            f"Tracked {len(tracks['players'])} players, {len(tracks['referees'])} referees")
        return tracks


class TeamAssigner:
    """Assign players to teams based on jersey color"""

    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        """Get KMeans clustering model for image"""
        from sklearn.cluster import KMeans

        # Reshape image to 2D array of pixels
        image_2d = image.reshape(-1, 3)

        # Perform KMeans with 2 clusters
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)

        return kmeans

    def get_player_color(self, frame, bbox):
        """Extract player jersey color"""
        x1, y1, x2, y2 = map(int, bbox)

        # Ensure bbox is within frame boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)

        if x2 <= x1 or y2 <= y1:
            return None

        # Crop player image
        player_img = frame[y1:y2, x1:x2]

        if player_img.size == 0:
            return None

        # Get top half (jersey area)
        top_half = player_img[0:int(player_img.shape[0] / 2), :]

        if top_half.size == 0:
            return None

        # Get clustering model
        kmeans = self.get_clustering_model(top_half)

        # Get cluster labels
        labels = kmeans.labels_

        # Reshape labels to image shape
        clustered_image = labels.reshape(top_half.shape[0], top_half.shape[1])

        # Get player cluster (corner pixels are likely background)
        corner_clusters = [
            clustered_image[0, 0],
            clustered_image[0, -1],
            clustered_image[-1, 0],
            clustered_image[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters),
                                 key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]

        return player_color

    def assign_team_color(self, frame, player_detections):
        """Assign team colors based on player jerseys"""
        from sklearn.cluster import KMeans

        player_colors = []

        for player_id, detection in player_detections.items():
            bbox = detection.get('bbox', None)
            if bbox is not None:
                player_color = self.get_player_color(frame, bbox)
                if player_color is not None:
                    player_colors.append(player_color)

        if len(player_colors) < 2:
            return

        # Cluster players into 2 teams
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        kmeans.fit(player_colors)

        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        """Get team assignment for a player"""
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        if player_color is None:
            return 1  # Default team

        team_id = self.kmeans.predict(player_color.reshape(1,-1))[0]
        team_id+=1

        self.player_team_dict[player_id] = team_id

        return team_id

    def assign_teams(self, tracks, frames):
        """Assign teams to all players"""
        # Find a frame with many players to assign team colors
        frame_with_most_players = 0
        max_players = 0

        for frame_num in range(min(10, len(frames))):  # Check first 10 frames
            player_count = sum(1 for player_track in tracks['players'].values()
                               if frame_num in player_track)
            if player_count > max_players:
                max_players = player_count
                frame_with_most_players = frame_num

        if max_players > 0:
            player_detections = {}
            for player_id, player_track in tracks['players'].items():
                if frame_with_most_players in player_track:
                    player_detections[player_id] = player_track[
                        frame_with_most_players]

            self.assign_team_color(frames[frame_with_most_players],
                                   player_detections)

        # Assign team to each player in each frame
        for frame_num, frame in enumerate(frames):
            for player_id, track in tracks['players'].items():
                if frame_num in track:
                    bbox = track[frame_num]['bbox']
                    team = self.get_player_team(frame, bbox, player_id)
                    tracks['players'][player_id][frame_num]['team'] = team

                    if team in self.team_colors:
                        tracks['players'][player_id][frame_num]['team_color'] = \
                        self.team_colors[team]


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


def get_center_of_bbox(bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_foot_position(bbox):
    """Get foot position (bottom center of bbox)"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def draw_ellipse(frame, bbox, color, track_id=None):
    """Draw ellipse around player"""
    y2 = int(bbox[3])
    x_center, _ = get_center_of_bbox(bbox)
    width = int(bbox[2] - bbox[0])

    cv2.ellipse(
        frame,
        center=(x_center, y2),
        axes=(int(width), int(0.35 * width)),  #minor and major axis of ellipse
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color,
        thickness=2,
        lineType=cv2.LINE_4
    )

    if track_id is not None:
        rectangle_width = 40
        rectangle_height = 20
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = y2 - rectangle_height // 2 + 15
        y2_rect = y2 + rectangle_height // 2 + 15

        cv2.rectangle(frame, (x1_rect, y1_rect), (x2_rect, y2_rect), color,
                      cv2.FILLED)

        x1_text = x1_rect + 12
        if track_id > 99:
            x1_text -= 10

        cv2.putText(
            frame,
            f"{track_id}",
            (x1_text, y1_rect + 15),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

    return frame


def draw_triangle(frame, bbox, color):
    """Draw triangle above object"""
    y = int(bbox[1])
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],
        [x - 10, y - 20],
        [x + 10, y - 20]
    ])
    cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED) # color filled triangle
    cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)      # black border for triangle

    return frame


def draw_team_ball_control(frame, tracks, current_frame):
    """Draw ball control statistics"""
    # Calculate ball control
    ball_position = None
    if 1 in tracks['ball'] and current_frame in tracks['ball'][1]:
        ball_bbox = tracks['ball'][1][current_frame]['bbox']
        ball_position = get_foot_position(ball_bbox)

    team_ball_control = {1: 0, 2: 0}

    if ball_position is not None:
        for player_id, player_track in tracks['players'].items():
            if current_frame in player_track:
                team = player_track[current_frame].get('team', 1)
                player_bbox = player_track[current_frame]['bbox']
                player_position = get_foot_position(player_bbox)

                distance = np.linalg.norm(
                    np.array(ball_position) - np.array(player_position))

                if distance < 70:
                    team_ball_control[team] = team_ball_control.get(team, 0) + 1

    # Draw overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    total_control = sum(team_ball_control.values())
    if total_control > 0:
        team1_pct = (team_ball_control[1] / total_control) * 100
        team2_pct = (team_ball_control[2] / total_control) * 100
    else:
        team1_pct = team2_pct = 0

    cv2.putText(frame, f"Team 1 Ball Control: {team1_pct:.2f}%", (1400, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, f"Team 2 Ball Control: {team2_pct:.2f}%", (1400, 950),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    return frame


def draw_annotations(frames, tracks, camera_movement):
    """Draw all annotations on frames"""
    output_frames = []

    print("Drawing annotations on frames...")
    for frame_num, frame in enumerate(frames):
        frame = frame.copy()

        # Draw players
        for player_id, player_track in tracks['players'].items():
            if frame_num in player_track:
                bbox = player_track[frame_num]['bbox']
                team = player_track[frame_num].get('team', 1)
                color = (0, 255, 0) if team == 1 else (255, 0, 0)

                frame = draw_ellipse(frame, bbox, color, player_id)

                # Draw speed and distance
                if 'speed' in player_track[frame_num]:
                    speed = player_track[frame_num]['speed']
                    distance = player_track[frame_num].get('distance', 0)

                    x_center, y2 = get_foot_position(bbox)
                    cv2.putText(frame, f"{speed:.2f} km/h",
                                (x_center - 40, y2 + 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                2)
                    cv2.putText(frame, f"{distance:.2f} m",
                                (x_center - 40, y2 + 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255),
                                2)

        # Draw referees
        for referee_id, referee_track in tracks['referees'].items():
            if frame_num in referee_track:
                bbox = referee_track[frame_num]['bbox']
                frame = draw_ellipse(frame, bbox, (255, 255, 0))

        # Draw ball
        if 1 in tracks['ball'] and frame_num in tracks['ball'][1]:
            bbox = tracks['ball'][1][frame_num]['bbox']
            frame = draw_triangle(frame, bbox, (0, 255, 255))

        # Draw camera movement
        if frame_num < len(camera_movement):
            cv2.putText(frame,
                        f"Camera Movement X: {camera_movement[frame_num][0]:.2f}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)
            cv2.putText(frame,
                        f"Camera Movement Y: {camera_movement[frame_num][1]:.2f}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                        2)

        # Draw ball control
        frame = draw_team_ball_control(frame, tracks, frame_num)

        output_frames.append(frame)

        if frame_num % 50 == 0:
            print(f"Annotated {frame_num}/{len(frames)} frames")

    return output_frames


def main():
    """Main execution function"""
    print("=" * 70)
    print("FOOTBALL ANALYSIS SYSTEM")
    print("=" * 70)

    # Read video
    print("\n[1/7] Reading video...")
    video_frames = VideoUtils.read_video('../videos/match.mp4')

    # Initialize tracker
    print("\n[2/7] Initializing tracker...")
    tracker = Tracker('../models/best.pt')

    # Get tracks (set read_from_stub=False for first run)
    print("\n[3/7] Getting object tracks...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=True,  # Set to True after first successful run
        stub_path='../stubs/track_stubs.pkl'
    )

    #interpolate ball position
    tracks["ball"] = tracker.interpolate_ball_position(tracks["ball"])

    # Add position to tracks
    print("\n[4/7] Adding positions to tracks...")
    for object_type, object_tracks in tracks.items():
        for object_id, object_track in object_tracks.items():
            for frame_num, track_info in object_track.items():
                bbox = track_info['bbox']
                if object_type == 'ball':
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                track_info['position'] = position

    # Camera movement estimator
    print("\n[5/7] Estimating camera movement...")
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])
    camera_movement = camera_movement_estimator.get_camera_movement(
        video_frames,
        read_from_stub=True,  # Set to True after first successful run
        stub_path='../stubs/camera_movement_stub.pkl'
    )

    camera_movement_estimator.adjust_positions_to_camera_movement(tracks,
                                                                  camera_movement)

    # View transformer
    print("\n[6/7] Transforming view and calculating metrics...")
    view_transformer = ViewTransformer()
    view_transformer.add_transformed_position_to_tracks(tracks)

    # Speed and distance
    speed_distance_estimator = SpeedAndDistanceEstimator()
    speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    # Team assignment
    team_assigner = TeamAssigner()
    team_assigner.assign_teams(tracks, video_frames)

    # Draw annotations
    print("\n[7/7] Drawing annotations...")
    output_frames = draw_annotations(video_frames, tracks, camera_movement)

    # Save video
    print("\nSaving output video...")
    os.makedirs('../output_videos', exist_ok=True)
    VideoUtils.save_video(output_frames, '../output_videos/output.mp4')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Players tracked: {len(tracks['players'])}")
    print(f"Referees tracked: {len(tracks['referees'])}")
    print(f"Output saved to: output_videos/output.mp4")


if __name__ == '__main__':
    main()