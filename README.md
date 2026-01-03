# Sistem Pengenalan Angka Tulisan Tangan Menggunakan K-Nearest Neighbor (KNN)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![MNIST](https://img.shields.io/badge/Dataset-MNIST-orange.svg)](http://yann.lecun.com/exdb/mnist/)

> **Tugas Besar - Mata Kuliah Kecerdasan Buatan**  
> Program Studi Teknik Komputer, Fakultas Teknik Elektro  
> Universitas Telkom, 2025

---

<div align="center">

**⭐ Star this repo if you find it helpful!**

</div>

Sistem ini merupakan implementasi algoritma **K-Nearest Neighbor (KNN)** untuk mengenali angka tulisan tangan (digit 0-9) menggunakan dataset **MNIST**. Project ini dirancang untuk memahami konsep dasar machine learning, khususnya algoritma klasifikasi berbasis jarak.

### Tujuan Project

1. Implementasi algoritma KNN manual (tanpa library ML tingkat tinggi)
2. Memahami preprocessing data untuk image recognition
3. Evaluasi performa model dengan berbagai metrik
4. Menguji model pada data eksternal (gambar tulisan tangan pengguna)

---

## Fitur Utama

### 1. **Implementasi KNN Manual**
- Perhitungan jarak Euclidean vectorized
- Voting mayoritas dengan tie-breaking
- Parameter K dapat disesuaikan

### 2. **Preprocessing MNIST-Style**
- Auto-invert background
- Otsu thresholding untuk binarization
- Bounding box detection & cropping
- Center of mass alignment
- Gaussian blur untuk noise reduction

### 3. **Evaluasi Komprehensif**
- Confusion Matrix visualization
- Metrics: Accuracy, Precision, Recall, F1-Score
- Analisis kesalahan per kelas
- Eksperimen berbagai nilai K

### 4. **Upload Gambar Interaktif**
- Support Google Colab (file upload widget)
- Support VS Code/Jupyter (file dialog GUI)
- Fallback manual path input
- Real-time preprocessing & prediction

### 5. **Visualisasi Lengkap**
- Sample dataset visualization
- Prediction comparison (True vs Predicted)
- Confusion matrix heatmap
- K-value performance comparison
- Before/after preprocessing display

---

## Teknologi

### Language & Tools
- **Python 3.8+**
- **Jupyter Notebook / VS Code**
- **Google Colab** (optional)

### Libraries & Dependencies
```
numpy >= 1.21.0          # Numerical computing
matplotlib >= 3.4.0      # Visualization
scikit-learn >= 1.0.0    # Dataset & metrics
pillow >= 8.3.0          # Image processing
seaborn >= 0.11.0        # Statistical visualization
joblib >= 1.0.0          # Model persistence
```

---

## Instalasi

### 1. Clone Repository
```bash
git clone https://github.com/username/handwritten-digit-knn.git
cd handwritten-digit-knn
```

### 2. Install Dependencies
```bash
pip install -q numpy matplotlib scikit-learn joblib pillow seaborn
```

### 3. Run Notebook
```bash
# VS Code
code Handwritten_Digit_Project_CLEAN.ipynb

# Jupyter
jupyter notebook Handwritten_Digit_Project_CLEAN.ipynb

# Google Colab
# Upload notebook ke Google Drive dan buka dengan Colab
```

---

## Cara Penggunaan

### Menjalankan Training & Evaluasi

1. **Open Notebook**
   ```
   Handwritten_Digit_Project_CLEAN.ipynb
   ```

2. **Run All Cells** (Ctrl+Shift+Enter)
   - Cell 1: Install dependencies
   - Cell 2-3: Load MNIST dataset
   - Cell 4-5: Split data (80/20)
   - Cell 6-7: Implement KNN
   - Cell 8-10: Evaluate model
   - Cell 11: Eksperimen nilai K
   - Cell 12: Save model

3. **Model Tersimpan**
   ```
   knn_pipeline.joblib (56,000 training samples, K=5)
   ```

### Testing dengan Gambar Eksternal

1. **Run Cell Pengujian Eksternal**

2. **Upload Gambar**
   - **Google Colab:** Klik tombol "Choose Files"
   - **VS Code/Jupyter:** File dialog akan terbuka otomatis
   - **Manual:** Input path file

3. **Lihat Hasil**
   - Gambar original
   - Setelah preprocessing (28×28)
   - Prediksi digit

---

## Struktur Project

```
HDG/
├── Handwritten_Digit_Project_CLEAN.ipynb    # Main notebook (bersih & terstruktur)
├── Handwritten_Digit_Project.ipynb          # Original notebook
├── knn_pipeline.joblib                      # Saved model
├── README.md                                # Documentation (you are here)
├── Image/                                   # Folder untuk test images (optional)
│   ├── digit_0.png
│   ├── digit_5.png
│   └── ...
└── requirements.txt                         # Dependencies list
```

### 1. Dataset: MNIST
- **Total:** 70,000 gambar (28×28 piksel grayscale)
- **Training:** 56,000 samples (80%)
- **Testing:** 2,000 samples (subset untuk efisiensi)
- **Classes:** 10 digit (0-9)

### 2. Algoritma: K-Nearest Neighbor (KNN)

#### Cara Kerja:
```
1. Hitung jarak Euclidean dari query point ke semua training points
2. Pilih K tetangga terdekat
3. Voting mayoritas dari label tetangga
4. Return label dengan vote terbanyak
```

#### Formula Jarak Euclidean:
```
d(x, y) = √(Σ(xi - yi)²)
```

### 3. Preprocessing Pipeline

```
Input Image
    ↓
Grayscale Conversion
    ↓
Auto-Invert (if bright background)
    ↓
Gaussian Blur (noise reduction)
    ↓
Otsu Thresholding (binarization)
    ↓
Border Noise Removal
    ↓
Bounding Box Detection & Crop
    ↓
Resize to 20×20 (preserve aspect ratio)
    ↓
Center on 28×28 Canvas
    ↓
Center of Mass Adjustment
    ↓
Final Smoothing
    ↓
Flatten to 784 Vector
```

### 4. Hyperparameter Tuning

Eksperimen dengan berbagai nilai K:

| K  | Akurasi | Keterangan |
|----|---------|------------|
| 3  | 96.8%   | Sensitif terhadap noise |
| **5**  | **97.6%**   | **Optimal ✓** |
| 7  | 97.4%   | Stabil |
| 11 | 97.0%   | Oversmoothing |

**Kesimpulan:** K=5 memberikan performa terbaik

---

### Metrik Evaluasi (K=5, Test Set)

| Metric    | Score   |
|-----------|---------|
| Akurasi   | 97.55%  |
| Precision | 97.56%  |
| Recall    | 97.55%  |
| F1-Score  | 97.55%  |

### Confusion Matrix

Model menunjukkan performa excellent dengan kesalahan minimal:
- **Digit paling akurat:** 0, 1, 6
- **Digit paling challenging:** 4 ↔ 9, 3 ↔ 8

### Akurasi Per Kelas

```
Digit 0: 98.5%
Digit 1: 99.1%
Digit 2: 97.2%
Digit 3: 96.8%
Digit 4: 96.5%
Digit 5: 97.1%
Digit 6: 98.2%
Digit 7: 97.3%
Digit 8: 96.4%
Digit 9: 96.7%
```

### Performance Analysis

**Kelebihan:**
- ✅ Akurasi tinggi (>97%) pada MNIST
- ✅ Implementasi sederhana & interpretable
- ✅ Tidak perlu training time
- ✅ Robust dengan preprocessing yang baik

**Kekurangan:**
- ⚠️ Lambat pada prediksi (O(n) per query)
- ⚠️ Memory intensive (store semua training data)
- ⚠️ Sensitif terhadap kualitas preprocessing
- ⚠️ Performa menurun pada gambar eksternal dengan kualitas rendah

---

## Tim Pengembang

| Nama                        | NIM            | Role                                |
|-----------------------------|----------------|-------------------------------------|
| **Ghavind Azzarya**         | 101032300012   | Pembuatan Program, Sample Test, Laporan |
| **Muammar Yaldi Fadhil**    | 101032300019   | Pembuatan Program, Sample Test      |
| **Ibnu Sugeng Mugiana**     | 101032300124   | Pembuatan Laporan, Presentasi       |

**Dosen Pengampu:**  
Dr. Budhi Irawan, S.Si., M.T

**Program Studi:**  
Teknik Komputer, Fakultas Teknik Elektro, Universitas Telkom

---

## Referensi

### Dataset
- LeCun, Y., Cortes, C., & Burges, C. (1998). **MNIST Handwritten Digit Database**. [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
- OpenML MNIST. [https://www.openml.org/d/554](https://www.openml.org/d/554)

### Algoritma
- Cover, T., & Hart, P. (1967). **Nearest neighbor pattern classification**. IEEE Transactions on Information Theory.
- Scikit-learn Documentation. **K-Nearest Neighbors**. [https://scikit-learn.org/](https://scikit-learn.org/)

### Image Processing
- Otsu, N. (1979). **A threshold selection method from gray-level histograms**. IEEE Transactions on Systems, Man, and Cybernetics.
- PIL Documentation. [https://pillow.readthedocs.io/](https://pillow.readthedocs.io/)

### Machine Learning
- Wikipedia. **K-nearest neighbors algorithm**. [https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
- IBM. **What is Artificial Intelligence?** [https://www.ibm.com/topics/artificial-intelligence](https://www.ibm.com/topics/artificial-intelligence)

---

## Pembelajaran & Insights

### Key Takeaways

1. **Preprocessing is Critical**
   - Kualitas preprocessing menentukan 50% keberhasilan
   - Center of mass alignment crucial untuk consistency
   - Otsu thresholding effective untuk binarization

2. **K Selection Matters**
   - K terlalu kecil → overfitting
   - K terlalu besar → undersmoothing
   - Sweet spot: K=5 untuk MNIST

3. **Trade-offs**
   - Simplicity vs Performance
   - Accuracy vs Speed
   - Memory vs Computation

### Future Improvements

1. **Algoritma:**
   - Implement CNN (>99% accuracy)
   - Try SVM, Random Forest
   - Ensemble methods

2. **Optimasi:**
   - KD-Tree untuk faster search
   - Dimensionality reduction (PCA)
   - GPU acceleration

3. **Features:**
   - Real-time webcam input
   - Multi-digit recognition
   - Mobile app deployment
   - Web interface

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- **Our Mentor** - Guidance & mentorship
- **MNIST Contributors** - Dataset availability
- **Open Source Community** - Tools & libraries

Made with passion by Tim Teknik Komputer
