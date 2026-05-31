import cv2
import numpy as np

from gesture import GestureDetector

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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    gesture_detector = GestureDetector(
        history_len=6,
        gesture_threshold=35,
        cooldown_frames=20
    )

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
        
        # Gambarkan tracker titik tengah ke frame RGB asli
        if centroid:
            cx, cy = centroid
            # Menambahkan lingkaran pointer + teks titik koordinat
            cv2.circle(frame, (cx, cy), 15, (0, 255, 0), -1)
            cv2.circle(frame, (cx, cy), 15, (255, 255, 255), 2)
            cv2.putText(frame, f"Blue Marker: ({cx},{cy})", (cx + 25, cy - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 180, 0), 2)

        velocity = gesture_detector.get_velocity()
        if gesture_detector.is_active():
            cv2.putText(frame, "GESTURE: FAST MOVE", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"Velocity: {velocity:.1f} px/frame", (20, 40),
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
