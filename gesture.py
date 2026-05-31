from collections import deque

import numpy as np


class GestureDetector:
    """
    Deteksi gesture sederhana berdasarkan kecepatan perpindahan centroid.

    Jika centroid bergerak melewati threshold dalam satu frame, gerakan
    dianggap sebagai gesture cepat. Cooldown dipakai agar gesture tidak
    terdeteksi terus-menerus pada frame berurutan.
    """

    def __init__(self, history_len=6, gesture_threshold=35, cooldown_frames=20):
        self.history = deque(maxlen=history_len)
        self.gesture_threshold = gesture_threshold
        self.cooldown_frames = cooldown_frames
        self.cooldown = 0
        self.current_velocity = 0.0
        self.gesture_active = False

    def update(self, centroid):
        if self.cooldown > 0:
            self.cooldown -= 1

        if centroid is None:
            self.history.clear()
            self.current_velocity = 0.0
            self.gesture_active = False
            return

        self.history.append(np.array(centroid, dtype=np.float32))
        if len(self.history) < 2:
            self.current_velocity = 0.0
            self.gesture_active = False
            return

        previous = self.history[-2]
        current = self.history[-1]
        self.current_velocity = float(np.linalg.norm(current - previous))

        if self.current_velocity > self.gesture_threshold and self.cooldown == 0:
            self.gesture_active = True
            self.cooldown = self.cooldown_frames
        else:
            self.gesture_active = False

    def is_active(self):
        return self.gesture_active

    def get_velocity(self):
        return self.current_velocity
