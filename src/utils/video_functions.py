import cv2
import numpy as np


def read_video(video_path):
    """
    Read video file and return list of frames

    Args:
        video_path: Path to video file

    Returns:
        List of frames (numpy arrays)
    """
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


def save_video(frames, output_path, fps=24):
    """
    Save frames to video file

    Args:
        frames: List of frames (numpy arrays)
        output_path: Path to save video
        fps: Frames per second
    """
    if len(frames) == 0:
        print("No frames to save!")
        return

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Define codec and create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved {len(frames)} frames to {output_path}")


def calculate_optical_flow(prev_frame, curr_frame, prev_points):
    """
    Calculate optical flow between two frames

    Args:
        prev_frame: Previous frame (grayscale)
        curr_frame: Current frame (grayscale)
        prev_points: Points to track from previous frame

    Returns:
        New points, status array
    """
    # Parameters for lucas kanade optical flow
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=2,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )

    # Calculate optical flow
    next_points, status, error = cv2.calcOpticalFlowPyrLK(
        prev_frame, curr_frame, prev_points, None, **lk_params
    )

    return next_points, status


def estimate_camera_movement(frames, n_features=100):
    """
    Estimate camera movement across frames using optical flow

    Args:
        frames: List of video frames
        n_features: Number of features to track

    Returns:
        List of camera movements (dx, dy) for each frame
    """
    camera_movements = [[0, 0]]  # First frame has no movement

    # Convert first frame to grayscale
    prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)

    # Detect features in first frame
    prev_points = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=n_features,
        qualityLevel=0.01,
        minDistance=10
    )

    for i in range(1, len(frames)):
        curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)

        # Calculate optical flow
        next_points, status = calculate_optical_flow(prev_gray, curr_gray,
                                                     prev_points)

        # Filter good points
        good_prev = prev_points[status == 1]
        good_next = next_points[status == 1]

        if len(good_prev) > 0:
            # Calculate average movement
            dx = np.median(good_next[:, 0, 0] - good_prev[:, 0, 0])
            dy = np.median(good_next[:, 0, 1] - good_prev[:, 0, 1])
            camera_movements.append([dx, dy])
        else:
            camera_movements.append([0, 0])

        # Update for next iteration
        prev_gray = curr_gray.copy()
        prev_points = cv2.goodFeaturesToTrack(
            prev_gray,
            maxCorners=n_features,
            qualityLevel=0.01,
            minDistance=10
        )

    return np.array(camera_movements)


def adjust_positions_for_camera_movement(positions, camera_movements):
    """
    Adjust object positions to account for camera movement

    Args:
        positions: List of (x, y) positions
        camera_movements: Camera movement for each frame

    Returns:
        Adjusted positions
    """
    adjusted_positions = []
    cumulative_movement = [0, 0]

    for i, pos in enumerate(positions):
        if i < len(camera_movements):
            cumulative_movement[0] -= camera_movements[i][0]
            cumulative_movement[1] -= camera_movements[i][1]

        adjusted_pos = [
            pos[0] + cumulative_movement[0],
            pos[1] + cumulative_movement[1]
        ]
        adjusted_positions.append(adjusted_pos)

    return adjusted_positions


def get_perspective_transform(src_points, dst_points):
    """
    Calculate perspective transformation matrix

    Args:
        src_points: Source points (4 corners in image)
        dst_points: Destination points (4 corners in real world)

    Returns:
        Transformation matrix
    """
    src_points = np.float32(src_points)
    dst_points = np.float32(dst_points)

    matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    return matrix


def apply_perspective_transform(points, matrix):
    """
    Apply perspective transformation to points

    Args:
        points: List of (x, y) points
        matrix: Transformation matrix

    Returns:
        Transformed points
    """
    # Convert points to homogeneous coordinates
    points = np.array(points, dtype=np.float32)
    if len(points.shape) == 1:
        points = points.reshape(1, -1)

    points_homogeneous = np.hstack([points, np.ones((points.shape[0], 1))])

    # Apply transformation
    transformed = np.dot(matrix, points_homogeneous.T).T

    # Convert back to 2D coordinates
    transformed_2d = transformed[:, :2] / transformed[:, 2:3]

    return transformed_2d


def pixel_to_meters(pixel_distance, reference_distance_pixels,
                    reference_distance_meters):
    """
    Convert pixel distance to meters

    Args:
        pixel_distance: Distance in pixels
        reference_distance_pixels: Known reference distance in pixels
        reference_distance_meters: Known reference distance in meters

    Returns:
        Distance in meters
    """
    scale = reference_distance_meters / reference_distance_pixels
    return pixel_distance * scale


def calculate_speed(positions, fps, pixel_to_meter_scale):
    """
    Calculate speed from position changes

    Args:
        positions: List of positions [(x, y), ...]
        fps: Frames per second of video
        pixel_to_meter_scale: Conversion factor from pixels to meters

    Returns:
        List of speeds in km/h
    """
    speeds = [0]  # First frame has no speed

    for i in range(1, len(positions)):
        # Calculate distance
        dx = positions[i][0] - positions[i - 1][0]
        dy = positions[i][1] - positions[i - 1][1]
        distance_pixels = np.sqrt(dx ** 2 + dy ** 2)

        # Convert to meters
        distance_meters = distance_pixels * pixel_to_meter_scale

        # Calculate speed (m/s to km/h)
        time_seconds = 1 / fps
        speed_mps = distance_meters / time_seconds
        speed_kmh = speed_mps * 3.6

        speeds.append(speed_kmh)

    return speeds


def draw_speed_and_distance(frame, track_id, position, speed, distance, color):
    """
    Draw speed and distance information on frame

    Args:
        frame: Video frame
        track_id: Object tracking ID
        position: (x, y) position
        speed: Speed in km/h
        distance: Total distance in meters
        color: Color for text

    Returns:
        Frame with annotations
    """
    x, y = map(int, position)

    # Draw text background
    text = f"ID:{track_id} Speed:{speed:.1f}km/h Dist:{distance:.1f}m"
    (text_width, text_height), _ = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    cv2.rectangle(
        frame,
        (x, y - text_height - 5),
        (x + text_width, y),
        color,
        -1
    )

    # Draw text
    cv2.putText(
        frame,
        text,
        (x, y - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return frame