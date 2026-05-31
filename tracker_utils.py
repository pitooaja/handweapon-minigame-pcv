import cv2
import numpy as np

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
