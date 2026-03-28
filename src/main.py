import os

from camera_movement import CameraMovementEstimator
from drawing import draw_annotations, get_center_of_bbox, get_foot_position
from speed_distance import SpeedAndDistanceEstimator
from team_assigner import TeamAssigner
from tracker import Tracker
from utils.video_utils import read_video, save_video
from view_transformer import ViewTransformer
from player_ball_assigner import PlayerBallAssigner


def main():
    """Main execution function"""
    print("=" * 70)
    print("FOOTBALL ANALYSIS SYSTEM")
    print("=" * 70)

    # Read video
    print("\n[1/7] Reading video...")
    video_frames = read_video('../videos/match.mp4')

    # Initialize tracker
    print("\n[2/7] Initializing tracker...")
    tracker = Tracker('../models/best (1).pt')

    # Get tracks (set read_from_stub=False for first run)
    print("\n[3/7] Getting object tracks...")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=False,  # Set to True after first successful run
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


    # Player ball assignment (closest player to ball per frame)
    player_ball_assigner = PlayerBallAssigner()
    possession_by_frame = []
    for frame_num in range(len(video_frames)):
        if 1 not in tracks["ball"] or frame_num not in tracks["ball"][1]:
            possession_by_frame.append(None)
            continue
        ball_bbox = tracks["ball"][1][frame_num]["bbox"]
        players_this_frame = {
            pid: track[frame_num]
            for pid, track in tracks["players"].items()
            if frame_num in track
        }
        closest_player = player_ball_assigner.assign_players_to_ball(
            players_this_frame, ball_bbox
        )
        if closest_player is not None:
            team = tracks["players"][closest_player][frame_num]["team"]
            tracks["players"][closest_player][frame_num]["has_ball"] = True
            possession_by_frame.append(team)
        else:
            possession_by_frame.append(None)

    # Draw annotations
    print("\n[7/7] Drawing annotations...")
    output_frames = draw_annotations(
        video_frames, tracks, camera_movement, possession_by_frame
    )

    # Save video
    print("\nSaving output video...")
    os.makedirs('../output_videos', exist_ok=True)
    save_video(output_frames, '../output_videos/output.mp4')

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)
    print(f"Players tracked: {len(tracks['players'])}")
    print(f"Referees tracked: {len(tracks['referees'])}")
    print(f"Output saved to: output_videos/output.mp4")


if __name__ == '__main__':
    main()
