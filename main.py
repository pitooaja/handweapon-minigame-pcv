import cv2
import numpy as np

from gesture import GestureDetector

FRAME_W = 640
FRAME_H = 480
ANVIL_X = FRAME_W // 2
ANVIL_Y = 360
INGOT_W = 150
INGOT_H = 42
HIT_FLASH_FRAMES = 10

def manual_erosion(binary_img, kernel_size=5):
    """
    Operasi Erosi murni NumPy (tanpa cv2.erode).
    Menyusutkan area putih. Semacam menggeser / stride pixel
    dan hanya akan aktif bila seluruh area bernilai True.
    """
    pad_size = kernel_size // 2
    padded = np.pad(binary_img, pad_size, mode='constant', constant_values=0)
    
    rows, cols = binary_img.shape
    eroded = np.ones((rows, cols), dtype=bool)
    
    padded_bool = padded == 255
    
    # Penggunaan array shifting menggantikan nested loop satu per satu (Jauh lebih cepat)
    for i in range(kernel_size):
        for j in range(kernel_size):
            shifted = padded_bool[i:i+rows, j:j+cols]
            eroded = eroded & shifted
            
    return (eroded.astype(np.uint8) * 255)

def manual_dilation(binary_img, kernel_size=5):
    """
    Operasi Dilasi murni NumPy (tanpa cv2.dilate).
    Memperbesar area putih. Menggeser area piksel dan
    menggabungkannya menggunakan operator bitwise OR.
    """
    pad_size = kernel_size // 2
    padded = np.pad(binary_img, pad_size, mode='constant', constant_values=0)
    
    rows, cols = binary_img.shape
    dilated = np.zeros((rows, cols), dtype=bool)
    
    padded_bool = padded == 255
    
    for i in range(kernel_size):
        for j in range(kernel_size):
            shifted = padded_bool[i:i+rows, j:j+cols]
            dilated = dilated | shifted
            
    return (dilated.astype(np.uint8) * 255)

def manual_opening(binary_img, kernel_size=5):
    """Opening menghilangkan noise di luar objek: Erosi -> Dilasi"""
    eroded = manual_erosion(binary_img, kernel_size)
    return manual_dilation(eroded, kernel_size)

def manual_closing(binary_img, kernel_size=5):
    """Closing menutupi lubang kecil di dalam objek: Dilasi -> Erosi"""
    dilated = manual_dilation(binary_img, kernel_size)
    return manual_erosion(dilated, kernel_size)

def get_centroid(binary_img):
    """
    Kalkulasi centroid manual dari citra/masker biner
    murni dengan manipulasi indeks dan fungsi rerata(mean) Numpy
    """
    y_indices, x_indices = np.nonzero(binary_img)
    if len(x_indices) > 500: # Threshold luasan minimal (agar noise kecil tidak dideteksi sebagai tangan)
        cx = int(np.mean(x_indices))
        cy = int(np.mean(y_indices))
        return cx, cy
    return None

def create_hammer_sprite(width=110, height=120):
    """
    Membuat sprite hammer sederhana dalam format BGRA.
    Channel alpha dipakai agar sprite bisa ditempel transparan ke frame.
    """
    sprite = np.zeros((height, width, 4), dtype=np.uint8)
    cx = width // 2

    # Handle palu.
    for y in range(34, height):
        x = int(cx + (y - 34) * 0.10)
        sprite[y, x - 5:x + 6, :3] = (45, 85, 150)
        sprite[y, x - 5:x + 6, 3] = 245
        sprite[y, x - 1:x + 2, :3] = (90, 145, 210)

    # Kepala palu.
    head_x1, head_y1 = cx - 38, 14
    head_x2, head_y2 = cx + 38, 44
    sprite[head_y1:head_y2, head_x1:head_x2, :3] = (150, 160, 170)
    sprite[head_y1:head_y2, head_x1:head_x2, 3] = 250
    sprite[head_y1 + 4:head_y1 + 10, head_x1 + 5:head_x2 - 5, :3] = (230, 235, 240)

    # Aura biru di sekitar marker agar hubungan marker-senjata terlihat.
    ys, xs = np.mgrid[0:height, 0:width]
    dist = np.sqrt((xs - cx) ** 2 + (ys - 78) ** 2)
    aura = dist <= 23
    sprite[:, :, 0] = np.where(aura, 255, sprite[:, :, 0])
    sprite[:, :, 1] = np.where(aura, 185, sprite[:, :, 1])
    sprite[:, :, 2] = np.where(aura, 50, sprite[:, :, 2])
    sprite[:, :, 3] = np.where(aura, np.maximum(sprite[:, :, 3], 95), sprite[:, :, 3])

    return sprite

def overlay_sprite(frame, sprite_bgra, cx, cy, scale=1.0):
    """
    Menempel sprite BGRA ke frame BGR dengan alpha blending manual NumPy.
    Rumus: output = foreground * alpha + background * (1 - alpha)
    """
    sprite = sprite_bgra
    if abs(scale - 1.0) > 0.01:
        sh, sw = sprite.shape[:2]
        new_w = max(1, int(sw * scale))
        new_h = max(1, int(sh * scale))
        sprite = cv2.resize(sprite, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    sh, sw = sprite.shape[:2]
    x1 = cx - sw // 2
    y1 = cy - sh // 2
    x2 = x1 + sw
    y2 = y1 + sh

    sx1 = max(0, -x1)
    sy1 = max(0, -y1)
    sx2 = sw - max(0, x2 - frame.shape[1])
    sy2 = sh - max(0, y2 - frame.shape[0])
    fx1 = max(0, x1)
    fy1 = max(0, y1)
    fx2 = fx1 + (sx2 - sx1)
    fy2 = fy1 + (sy2 - sy1)

    if fx2 <= fx1 or fy2 <= fy1:
        return

    roi = frame[fy1:fy2, fx1:fx2].astype(np.float32)
    sprite_crop = sprite[sy1:sy2, sx1:sx2]
    foreground = sprite_crop[:, :, :3].astype(np.float32)
    alpha = sprite_crop[:, :, 3:4].astype(np.float32) / 255.0

    blended = foreground * alpha + roi * (1.0 - alpha)
    frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)

def draw_anvil(frame):
    """Menggambar anvil sebagai area target utama Forge Strike."""
    base_y = ANVIL_Y + 28
    cv2.ellipse(frame, (ANVIL_X, base_y), (160, 22), 0, 0, 360, (35, 35, 40), -1)
    cv2.rectangle(frame, (ANVIL_X - 120, ANVIL_Y), (ANVIL_X + 120, ANVIL_Y + 38),
                  (70, 75, 85), -1)
    cv2.rectangle(frame, (ANVIL_X - 82, ANVIL_Y + 38), (ANVIL_X + 82, ANVIL_Y + 85),
                  (50, 55, 65), -1)
    cv2.rectangle(frame, (ANVIL_X - 150, ANVIL_Y + 85), (ANVIL_X + 150, ANVIL_Y + 105),
                  (42, 44, 50), -1)
    cv2.line(frame, (ANVIL_X - 112, ANVIL_Y + 8), (ANVIL_X + 112, ANVIL_Y + 8),
             (150, 155, 165), 2)
    cv2.putText(frame, "ANVIL", (ANVIL_X - 29, ANVIL_Y + 64),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (190, 190, 195), 1)

def draw_ingot(frame, hit_flash=0):
    """Menggambar hot ingot sebagai second object yang akan dipukul."""
    x1 = ANVIL_X - INGOT_W // 2
    y1 = ANVIL_Y - 54
    x2 = ANVIL_X + INGOT_W // 2
    y2 = y1 + INGOT_H

    glow_color = (0, 180, 255) if hit_flash == 0 else (0, 255, 255)
    body_color = (0, 110, 230) if hit_flash == 0 else (40, 220, 255)

    cv2.rectangle(frame, (x1 - 8, y1 - 8), (x2 + 8, y2 + 8), glow_color, 2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), body_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 230, 150), 2)
    cv2.putText(frame, "INGOT", (x1 + 44, y1 + 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

    if hit_flash > 0:
        cv2.putText(frame, "HIT!", (ANVIL_X - 34, y1 - 18),
                    cv2.FONT_HERSHEY_DUPLEX, 0.85, (0, 255, 255), 2)

    return x1, y1, x2, y2

def is_strike_on_ingot(centroid, ingot_box, gesture_detector):
    """Cek apakah gesture STRIKE terjadi dekat second object."""
    if centroid is None or not gesture_detector.is_strike():
        return False

    cx, cy = centroid
    x1, y1, x2, y2 = ingot_box
    margin = 45
    return (x1 - margin <= cx <= x2 + margin and
            y1 - margin <= cy <= y2 + margin)

def dummy_callback(value):
    # Pass untuk trackbar
    pass

def main():
    # Default HSV untuk marker biru terang.
    # Bisa disesuaikan lewat trackbar jika pencahayaan berbeda.
    default_h_min, default_h_max = 90, 130
    default_s_min, default_s_max = 70, 255
    default_v_min, default_v_max = 50, 255

    # Inisiasi kalibrasi UI
    cv2.namedWindow("HSV Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HSV Calibration", 400, 350)
    
    # OpenCV under Windows sometimes requires an image to render trackbars on properly 
    # Gunakan gambar latar kosong yang cukup tinggi agar area trackbar punya ruang untuk muncul
    cv2.imshow("HSV Calibration", np.zeros((100, 400, 3), np.uint8))
    
    # Nilai standar untuk mendeteksi marker biru
    cv2.createTrackbar("H Min", "HSV Calibration", default_h_min, 179, dummy_callback)
    cv2.createTrackbar("H Max", "HSV Calibration", default_h_max, 179, dummy_callback)
    cv2.createTrackbar("S Min", "HSV Calibration", default_s_min, 255, dummy_callback)
    cv2.createTrackbar("S Max", "HSV Calibration", default_s_max, 255, dummy_callback)
    cv2.createTrackbar("V Min", "HSV Calibration", default_v_min, 255, dummy_callback)
    cv2.createTrackbar("V Max", "HSV Calibration", default_v_max, 255, dummy_callback)

    # Inisialisasi Kamera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    
    gesture_detector = GestureDetector(
        history_len=6,
        gesture_threshold=35,
        cooldown_frames=20
    )
    hammer_sprite = create_hammer_sprite()
    hit_flash = 0

    print("Mulai kamera. Arahkan marker biru ke kamera. Tekan 'q' untuk keluar.")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Flip layar agar sesuai efek cermin
        frame = cv2.flip(frame, 1)
        
        # 1. Konversi ke HSV
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        try:
            # 2. Ambil nilai range warna terbaru dari UI Trackbar
            h_min = cv2.getTrackbarPos("H Min", "HSV Calibration")
            h_max = cv2.getTrackbarPos("H Max", "HSV Calibration")
            s_min = cv2.getTrackbarPos("S Min", "HSV Calibration")
            s_max = cv2.getTrackbarPos("S Max", "HSV Calibration")
            v_min = cv2.getTrackbarPos("V Min", "HSV Calibration")
            v_max = cv2.getTrackbarPos("V Max", "HSV Calibration")
        except cv2.error:
            # Menghindari crash jika user menekan tombol 'Silang (Close)' dari window Trackbar secara manual
            break
        
        # Membuat mask biner murni dengan NumPy (tanpa cv2.inRange)
        lower_hsv = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper_hsv = np.array([h_max, s_max, v_max], dtype=np.uint8)
        
        mask_boolean = (
            (hsv_frame[:,:,0] >= lower_hsv[0]) & (hsv_frame[:,:,0] <= upper_hsv[0]) &
            (hsv_frame[:,:,1] >= lower_hsv[1]) & (hsv_frame[:,:,1] <= upper_hsv[1]) &
            (hsv_frame[:,:,2] >= lower_hsv[2]) & (hsv_frame[:,:,2] <= upper_hsv[2])
        )
        mask_mentah = mask_boolean.astype(np.uint8) * 255
        
        # 3. Operasi Morfologi Numerik Murni untuk Membersihkan Gambar
        # Kita downscale resolusinya terlebih dahulu menjadi setengah
        # Untuk menyeimbangkan proses perhitungannya yang full pure Numpy (dapat berat di resolusi penuh) 
        scale = 0.5 
        w = int(mask_mentah.shape[1] * scale)
        h = int(mask_mentah.shape[0] * scale)
        mask_small = cv2.resize(mask_mentah, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Melakukan Opening (membuang titik noise kecil di luar marker)
        mask_opened = manual_opening(mask_small, kernel_size=5)
        # Melakukan Closing (menambal lubang kecil di bagian dalam marker)
        mask_cleaned_small = manual_closing(mask_opened, kernel_size=5)
        
        # Kembalikan ukurannya ke normal untuk visualisasi
        mask_cleaned = cv2.resize(mask_cleaned_small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
        
        # 4. Cari Titik Tengah (Centroid) Menggunakan Numpy Array Indexing
        centroid = get_centroid(mask_cleaned)
        gesture_detector.update(centroid)

        draw_anvil(frame)
        ingot_box = draw_ingot(frame, hit_flash)
        if is_strike_on_ingot(centroid, ingot_box, gesture_detector):
            hit_flash = HIT_FLASH_FRAMES
        elif hit_flash > 0:
            hit_flash -= 1
        
        # Gambarkan tracker titik tengah ke frame RGB asli
        if centroid:
            cx, cy = centroid
            weapon_scale = 1.15 if gesture_detector.is_strike() else 1.0
            overlay_sprite(frame, hammer_sprite, cx, cy, scale=weapon_scale)

            # Menambahkan lingkaran pointer + teks titik koordinat
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
            cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
            cv2.putText(frame, f"Blue Marker: ({cx},{cy})", (cx + 25, cy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        velocity = gesture_detector.get_velocity()
        down_velocity = gesture_detector.get_down_velocity()
        if gesture_detector.is_strike():
            cv2.putText(frame, "GESTURE: STRIKE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Velocity: {velocity:.1f} | Down: {down_velocity:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        
        # 5. Result Display
        # Menampilkan gambar masing-masing ke dalam jendela layar
        cv2.imshow("1. Input Frame & Blue Marker Tracking", frame)
        cv2.imshow("2. Blue Mask Numpy Mentah", mask_mentah)
        cv2.imshow("3. Blue Mask Setelah Morfologi", mask_cleaned)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
