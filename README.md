# Forge Strike - Handweapon Mini Game PCV

Forge Strike adalah mini game berbasis computer vision menggunakan Python, OpenCV, dan NumPy. Marker biru digunakan sebagai hammer virtual. Pemain melakukan gesture `STRIKE` ke bawah untuk menempa hot ingot di atas anvil.

Project ini dikembangkan untuk memenuhi tugas mata kuliah Pengolahan Citra Visual / Computer Vision.

## Features

- Webcam capture real-time menggunakan `cv2.VideoCapture`.
- HSV color segmentation untuk mendeteksi marker biru.
- Binary mask dibuat manual dengan operasi array NumPy, tanpa `cv2.inRange`.
- Operasi morfologi manual: erosion, dilation, opening, dan closing.
- Centroid tracking sebagai posisi hammer virtual.
- Gesture detection `STRIKE` dari gerakan marker cepat ke bawah.
- Hammer weapon sprite overlay dengan alpha blending manual NumPy.
- Second object berupa hot ingot di atas anvil.
- Forge progress: ingot harus dipukul 3 kali sampai `FORGED`.
- Scoring system, hit counter, lives, timer ingot, start screen, dan game over.

## Project Structure

```text
handweapon-minigame-pcv/
|-- main.py           # Main game loop, rendering, HSV tracking, and game logic
|-- gesture.py        # Downward STRIKE gesture detector
|-- tracker_utils.py  # Manual morphology and centroid helper
|-- README.md         # Project documentation
`-- .gitignore
```

## Installation

```bash
pip install opencv-python numpy
```

## Run

```bash
python main.py
```

## Controls

- `SPACE`: start game
- `R`: restart after game over
- `q`: quit

## How to Play

1. Siapkan marker biru terang, misalnya layar HP dengan background biru, kertas biru, atau sarung tangan biru.
2. Arahkan marker ke kamera.
3. Tekan `SPACE` untuk mulai.
4. Gerakkan marker cepat ke bawah untuk melakukan `STRIKE`.
5. Arahkan hammer ke ingot di atas anvil.
6. Setiap hit menambah skor.
7. Tiga hit akan menyelesaikan ingot dan memberi bonus `FORGED`.
8. Jika timer ingot habis, lives berkurang.

## Default HSV Marker

```python
H_MIN, H_MAX = 90, 130
S_MIN, S_MAX = 70, 255
V_MIN, V_MAX = 50, 255
```

Nilai HSV masih bisa diatur melalui trackbar ketika program berjalan.

## Computer Vision Pipeline

1. Frame dibaca dari webcam.
2. Frame di-flip agar terasa seperti cermin.
3. Frame dikonversi dari BGR ke HSV.
4. Mask marker biru dibuat dengan boolean indexing NumPy.
5. Mask dibersihkan dengan manual opening dan closing.
6. Centroid marker dihitung dari mask biner.
7. Velocity centroid dipakai untuk mendeteksi downward `STRIKE`.
8. Hammer sprite ditempel ke posisi centroid dengan alpha blending manual.
9. Game logic mengecek hit ke ingot, score, lives, dan game state.

## Project Requirements Mapping

- Handweapon Mini Game: marker biru menjadi hammer virtual.
- Gesture Detection: downward `STRIKE` dari velocity centroid.
- Second Object: hot ingot di atas anvil.
- Scoring: score, hit counter, dan forged bonus.
- OpenCV I/O: `cv2.VideoCapture`, `cv2.imshow`, dan `cv2.waitKey`.
- HSV Segmentation: deteksi marker biru pada ruang warna HSV.
- NumPy Pixel Manipulation: threshold mask dan alpha blending manual.
- Manual Morphology: erosion, dilation, opening, dan closing.
- Weapon Sprite Overlay: hammer sprite BGRA ditempel ke frame kamera.

## Demo Checklist

- Screenshot start screen
- Screenshot gameplay
- Screenshot mask HSV
- Screenshot game over
- Link video demonstration
