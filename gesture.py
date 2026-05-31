from collections import deque

import numpy as np


class GestureDetector:
    """
    Deteksi gesture strike berdasarkan kecepatan perpindahan centroid.

    Untuk Forge Strike, gesture utama adalah gerakan cepat ke bawah.
    Cooldown dipakai agar strike tidak terdeteksi terus-menerus pada
    frame berurutan.
    """

    def __init__(self, history_len=6, gesture_threshold=35, cooldown_frames=20):
        self.history = deque(maxlen=history_len)
        self.gesture_threshold = gesture_threshold
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0
        self.current_velocity = 0.0
        self.down_velocity = 0.0
        self.gesture_active = False

    def update(self, centroid):
        if self.cooldown > 0:
            self.cooldown -= 1

        if centroid is None:
            self.history.clear()
            self.current_velocity = 0.0
            self.down_velocity = 0.0
            self.gesture_active = False
            return

        self.history.append(np.array(centroid, dtype=np.float32))
        if len(self.history) < 2:
            self.current_velocity = 0.0
            self.down_velocity = 0.0
            self.gesture_active = False
            return

        previous = self.history[-2]
        current = self.history[-1]
        delta = current - previous
        self.current_velocity = float(np.linalg.norm(delta))
        self.down_velocity = float(delta[1])

        mostly_down = self.down_velocity > abs(float(delta[0])) * 0.8
        fast_enough = self.down_velocity > self.gesture_threshold

        if mostly_down and fast_enough and self.cooldown == 0:
            self.gesture_active = True
            self.cooldown = self.cooldown_frames
        else:
            self.gesture_active = False

    def is_active(self):
        return self.gesture_active

    def is_strike(self):
        return self.gesture_active

    def get_velocity(self):
        return self.current_velocity

    def get_down_velocity(self):
        return self.down_velocity
