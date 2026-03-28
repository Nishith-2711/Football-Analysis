import cv2

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


def save_video(frames, output_path, fps=24):
    if len(frames) == 0:
        print("No frames to save!")
        return

    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Saved video to {output_path}")