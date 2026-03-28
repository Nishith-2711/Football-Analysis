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

    def get_player_team(self, frame, player_bbox, player_id,
                         is_goalkeeper=False):
        """Get team assignment for a player.

        Args:
            is_goalkeeper: True when the model classified this detection as
                           'goalkeeper'. Goalkeepers are always assigned to
                           team 1 regardless of jersey colour clustering.
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        # Goalkeeper detected by the model → always team 1
        if is_goalkeeper:
            self.player_team_dict[player_id] = 1
            return 1

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
                    is_gk = track[frame_num].get('is_goalkeeper', False)
                    team = self.get_player_team(frame, bbox, player_id,
                                                is_goalkeeper=is_gk)
                    tracks['players'][player_id][frame_num]['team'] = team

                    if team in self.team_colors:
                        tracks['players'][player_id][frame_num]['team_color'] = \
                        self.team_colors[team]
