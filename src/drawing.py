import cv2
import numpy as np


def get_center_of_bbox(bbox):
    """Get center point of bounding box"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_foot_position(bbox):
    """Get foot position (bottom center of bbox)"""
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1,p2):
    """Measure distance between 2 points"""
    distance = (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2
    return np.sqrt(distance)


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


def draw_team_ball_control(frame, tracks, frame_num, possession_by_frame=None):
    """Draw ball control statistics.

    If ``possession_by_frame`` is a list (team id 1 or 2 per frame, optional
    None), show **cumulative** share up to ``frame_num``. Otherwise use
    **per-frame** proximity to the ball (same as original).
    """
    # Draw overlay
    overlay = frame.copy()
    cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)
    alpha = 0.4
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if possession_by_frame is not None and len(possession_by_frame) > frame_num:
        so_far = possession_by_frame[: frame_num + 1]
        counted = [t for t in so_far if t in (1, 2)]
        team1_n = sum(1 for t in counted if t == 1)
        team2_n = sum(1 for t in counted if t == 2)
        total = team1_n + team2_n
        if total > 0:
            team1_pct = (team1_n / total) * 100
            team2_pct = (team2_n / total) * 100
        else:
            team1_pct = team2_pct = 0.0
    else:
        # Instantaneous: who is near the ball this frame
        ball_position = None
        if 1 in tracks["ball"] and frame_num in tracks["ball"][1]:
            ball_bbox = tracks["ball"][1][frame_num]["bbox"]
            ball_position = get_foot_position(ball_bbox)

        team_ball_control_dict = {1: 0, 2: 0}
        if ball_position is not None:
            for player_id, player_track in tracks["players"].items():
                if frame_num in player_track:
                    team = player_track[frame_num].get("team", 1)
                    player_bbox = player_track[frame_num]["bbox"]
                    player_position = get_foot_position(player_bbox)
                    distance = np.linalg.norm(
                        np.array(ball_position) - np.array(player_position)
                    )
                    if distance < 70:
                        team_ball_control_dict[team] = (
                            team_ball_control_dict.get(team, 0) + 1
                        )

        total_control = sum(team_ball_control_dict.values())
        if total_control > 0:
            team1_pct = (team_ball_control_dict[1] / total_control) * 100
            team2_pct = (team_ball_control_dict[2] / total_control) * 100
        else:
            team1_pct = team2_pct = 0.0

    cv2.putText(frame, f"Team 1 Ball Control: {team1_pct:.2f}%", (1400, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    cv2.putText(frame, f"Team 2 Ball Control: {team2_pct:.2f}%", (1400, 950),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

    return frame


def draw_annotations(frames, tracks, camera_movement, possession_by_frame=None):
    """Draw all annotations on frames.

    ``possession_by_frame``: optional list aligned with frames (team 1/2 when
    that team has the ball); used for cumulative ball-control overlay.
    """
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

                if player_track[frame_num].get('has_ball', False):
                    frame = draw_triangle(frame, bbox, (0, 0, 255))

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
        frame = draw_team_ball_control(
            frame, tracks, frame_num, possession_by_frame
        )

        output_frames.append(frame)

        if frame_num % 50 == 0:
            print(f"Annotated {frame_num}/{len(frames)} frames")

    return output_frames
