# Handweapon Mini Game PCV

Prototype mini game berbasis deteksi tangan menggunakan Python, OpenCV, dan NumPy untuk mata kuliah Pengolahan Citra Visual / Computer Vision.

Project ini masih berada pada tahap awal, yaitu membangun pipeline computer vision untuk membaca kamera, mendeteksi area tangan/marker dengan HSV, membersihkan mask, dan mengambil titik tengah objek sebagai dasar kontrol game.

## Current Progress

- Webcam capture menggunakan `cv2.VideoCapture`.
- Tampilan frame real-time menggunakan `cv2.imshow`.
- Konversi frame dari BGR ke HSV.
- HSV masking menggunakan operasi array NumPy.
- Operasi morfologi manual untuk membersihkan mask.
- Deteksi centroid sebagai posisi awal kontrol handweapon.

## Project Requirements

Mini game final harus memiliki:

- Gesture detection
- Second object
- Scoring system
- Webcam input dengan OpenCV
- Segmentasi warna HSV
- Manipulasi piksel dengan NumPy
- Operasi morfologi manual
- Weapon sprite overlay
- Dokumentasi dan source code di GitHub

## Project Structure

```text
handweapon-minigame-pcv/
|-- main.py           # Prototype kamera, HSV mask, morfologi, dan tracking
|-- tracker_utils.py  # Helper morfologi manual dan centroid
|-- README.md         # Dokumentasi project
`-- .gitignore
```

## Installation

Install dependencies:

```bash
pip install opencv-python numpy
```

## Run Prototype

```bash
python main.py
```

Tekan `q` untuk keluar dari program.

## Next Development Plan

- Menambahkan gesture detection dari pergerakan centroid.
- Menambahkan weapon virtual.
- Menambahkan second object sebagai target game.
- Menambahkan score, lives, dan game state.
- Mengembangkan konsep game final: Forge Strike.

