# player_ball_assigner.py

import numpy as np

from drawing import get_center_of_bbox, get_foot_position


class PlayerBallAssigner:
    """Assign which player is closest to the ball (pixel space)."""

    def __init__(self):
        # Pixels — tune to your resolution; slightly generous for interpolated boxes
        self.max_player_distance = 95

    def assign_players_to_ball(self, players, ball_bbox):
        """Return track id of closest player to the ball, or None."""
        if not players:
            return None

        bx1, by1, bx2, by2 = [float(x) for x in ball_bbox]
        ball_center = get_center_of_bbox((bx1, by1, bx2, by2))
        ball_feet = get_foot_position((bx1, by1, bx2, by2))

        min_distance = float("inf")
        closest_player = None

        for player_id, player in players.items():
            pb = player["bbox"]
            px1, py1, px2, py2 = [float(x) for x in pb]

            # Check if ball center falls inside the player's bounding box
            # (handles ball near torso / chest area, not just feet)
            if px1 <= ball_center[0] <= px2 and py1 <= ball_center[1] <= py2:
                distance = 0.0
            else:
                # Fallback: distance from ball to player feet
                player_feet = get_foot_position((px1, py1, px2, py2))
                d_center = np.linalg.norm(
                    np.array(ball_center, dtype=np.float64)
                    - np.array(player_feet, dtype=np.float64)
                )
                d_feet = np.linalg.norm(
                    np.array(ball_feet, dtype=np.float64)
                    - np.array(player_feet, dtype=np.float64)
                )
                distance = min(d_center, d_feet)

            if distance < self.max_player_distance and distance < min_distance:
                min_distance = distance
                closest_player = player_id

        return closest_player
