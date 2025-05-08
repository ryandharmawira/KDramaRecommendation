# Laporan Proyek Machine Learning - Ryan Dharmawira

## Project Overview

### Latar Belakang

Drama Korea (K-Drama) telah menjadi salah satu bentuk hiburan paling digemari secara global, dengan pertumbuhan penonton yang pesat di berbagai negara. Popularitas ini didorong oleh cerita yang kuat, produksi berkualitas tinggi, dan distribusi luas melalui platform streaming seperti Netflix dan Viu. Namun, dengan semakin banyaknya pilihan K-Drama, pengguna sering kali mengalami kebingungan dalam memilih tontonan yang sesuai dengan minat mereka.

Berdasarkan permasalahan ini, proyek ini menghadirkan sebuah sistem rekomendasi sebagai solusi untuk menemukan K-Drama yang relevan bagi pengguna secara efisien. Sistem rekomendasi dapat dikembangkan dengan dua pendekatan utama, yaitu content-based filtering dan collaborative filtering. Dalam proyek ini, digunakan pendekatan content-based filtering untuk merekomendasikan drama kepada pengguna dengan menganalisis kesamaan fitur-fitur konten, seperti genre sehingga sistem dapat menyarankan drama yang mirip dengan tontonan pengguna yang disukai sebelumnya atau preferensi pengguna.

## Business Understanding

### Problem Statements

- Pengguna platform streaming sering kesulitan menemukan K-Drama yang sesuai dengan preferensi mereka karena banyaknya pilihan.
- Penyedia layanan digital berisiko kehilangan pengguna jika tidak mampu menyajikan konten yang relevan secara efisien.

### Goals

- Mengembangkan sistem rekomendasi K-Drama yang dapat menyarankan konten berdasarkan minat atau perilaku pengguna.
- Meningkatkan retensi pengguna serta waktu tonton melalui sistem rekomendasi yang cerdas.

### Solution Statements

Untuk mencapai Goals tersebut, proyek ini membangun sebuah sistem rekomendasi menggunakan content-based filtering, yaitu dengan menganalisis fitur konten dari setiap drama untuk menentukan kemiripan antar item. Adapun metode yang akan dieksplorasi antara lain:
- TF-IDF pada Genre
- Genre Embedding dengan Pembobotan Rating


## Data Understanding

Dataset yang digunakan dalam proyek ini merupakan data yang berisi 350 daftar K-Drama.
Sumber dataset: [Kaggle - Korean Dramas Dataset](https://www.kaggle.com/datasets/saikalbatyrbekova/korean-dramas-dataset-eda).

### Variabel Dataset
- Rank : peringkat drama.
- Title : nama K-Drama.
- Number of Episodes : jumlah episode K-Drama.
- Year of release : tahun rilis K-Drama.
- Rating : peringkat berdasarkan penilaian penonton.
- Description : sinopsis singkat K-Drama.
- Genre : daftar genre/tema K-Drama.
- Tags : daftar kata kunci yang mendeskripsikan tema spesifik K-Drama.
- Actors : daftar pemeran K-Drama.

### Analisis Data

**Informasi Umum**
```
kdramas.info()
```
![Screenshot 2025-05-08 083323](https://github.com/user-attachments/assets/322efb86-18cc-454e-b123-b3ab1a11727d)

Dataset memiliki 350 data dan 7 atribut. Dari ketujuh atribut tersebut, terdapat 3 numerik dan 4 text.

**Pengecekan Missing Values**
```
kdramas.isnull().sum()
```
![Screenshot 2025-05-08 083346](https://github.com/user-attachments/assets/eeb40409-12a7-411b-a23d-916b155c0033)


Tidak ada missing values pada dataset.

**Visualisasi Distribusi Rating K-Drama**

![Screenshot 2025-05-08 083526](https://github.com/user-attachments/assets/d43d8542-75c4-4be5-906a-24b16dbda617)

Rating berkisar di sekitar 8.5 yang menandakan penonton secara umum merasa puas dengan K-Drama.

**Visualisasi Tren Perilisan K-Drama Tahun ke Tahun**

![Screenshot 2025-05-08 083539](https://github.com/user-attachments/assets/f45a96a2-d7c2-4ff0-85b9-ab94c2c45646)

Secara umum, terlihat K-Drama yang dirilis dari tahun ke tahun semakin meningkat. Hal ini dikarenakan adanya peningkatan produksi terutama pada 5 tahun terakhir yang menandakan ketertarikan masyarakat terhadap K-Drama meningkat.

**Visualisasi Top Genre K-Drama**

![Screenshot 2025-05-08 083554](https://github.com/user-attachments/assets/d610b58b-fc60-42af-a0b5-881b00dda26d)

Genre yang paling banyak ditonton adalah genre Drama dan Romance. Ini menandakan preferensi penonton lebih mengarah ke cerita hubungan asmara dan emosional. Di sisi lain, penonton hampir tidak tertarik dengan genre Sitcom, School dan Food.


## Data Preparation

Tahapan data preparation merupakan proses penting dalam membangun model machine learning, termasuk sistem rekomendasi. Pada proyek ini, Data Preparation dilakukan untuk memastikan bahwa data bersih, relevan, dan dalam format yang sesuai untuk digunakan dalam pemodelan content-based filtering. Sebelumnya pada bagian Data Understanding, terdapat pengecekan missing values. Karena tidak ada missing values, maka penanganan missing values tidak perlu dilakukan. Proses dilanjutkan dengan seleksi atribut yang hanya diperlukan saja yaitu Title, Rating, dan Genre. Hal ini dilakukan agar proses modeling lebih fokus dan efisien. Kemudian, dilakukan pembersihan atribut Genre. Atribut Genre diubah menjadi huruf kecil dan spasi dihapus untuk mempermudah pemrosesan teks. Hal ini dilakukan agar Genre bisa digunakan dalam teknik teks TF-IDF dan agar tidak terjadi ketidaksesuaian format. Langkah selanjutnya adalah mengkonversi Genre menjadi list. Genre dipisah berdasarkan koma menjadi list untuk keperluan multi-hot encoding pada pembuatan model pendekatan kedua. Hal ini dibutuhkan agar Genre dapat digunakan dalam encoding vektor dan pembobotan. Rating juga dilakukan normalisasi untuk pembobotan vektor secara lebih stabil untuk menghindari dominasi nilai Rating yang terlalu tinggi pada perhitungan. Hasil akhir dataset dari proses-proses ini dapat dilihat pada gambar berikut.

![Screenshot 2025-05-08 095703](https://github.com/user-attachments/assets/397477b4-3c3c-477d-a946-1fb25018d6ee)


## Modeling

### TF-IDF pada Genre

Metode ini menggunakan TF-IDF Vectorization terhadap fitur Genre K-Drama. Setiap K-Drama diubah menjadi representasi vektor berdasarkan Genre-nya. Kemudian, digunakan cosine similarity untuk mengukur kemiripan antar K-Drama. Adapun tahapan yang dilakukan sebagai berikut:
1. Melakukan vectorization dengan TfidfVectorizer.
2. Menghitung similarity antar K-Drama menggunakan cosine_similarity.
3. Mengambil Top-N K-Drama paling mirip sebagai rekomendasi.

```
# Mengambil salah satu Title acak dari dataframe
kdrama_title = df['Title'].sample(1).iloc[0]
print("Rekomendasi untuk", kdrama_title)
recommend_kdrama_tfidf(kdrama_title)
```

![Screenshot 2025-05-08 130449](https://github.com/user-attachments/assets/810ec599-0d49-4c51-bd00-7c1039039d53)

Metode ini memiliki beberapa kelebihan antara lain data interaksi pengguna tidak diperlukan dan hasil rekomendasi dapat dijelaskan dengan fitur konten (Genre). Di sisi lain juga terdapat beberapa kekurangan yaitu terbatas pada informasi yang ada dalam K-Drama itu sendiri yang berarti terdapat masalah cold-start untuk K-Drama baru dengan Genre unik.

### Genre Embedding dengan Pembobotan Rating

Metode ini menggunakan pendekatan serupa dengan metode yang sebelumnya, namun dengan pembobotan berdasarkan Rating. Genre diubah menjadi representasi biner/embedding, kemudian dilakukan pembobotan berdasarkan rating, agar Genre dari K-Drama dengan Rating tinggi mendapatkan bobot lebih besar dalam menentukan kemiripan. Adapun tahapan yang dilakukan sebagai berikut:
1. Mengubah fitur Genre ke dalam bentuk multi-hot encoding.
2. Melakukan pembobotan berdasarkan rating.
3. Menghitung cosine similarity antar vektor yang telah dibobot.
4. Menghasilkan rekomendasi berdasarkan skor kemiripan tertinggi.

```
# Mengambil salah satu Title acak dari dataframe
kdrama_title = df['Title'].sample(1).iloc[0]
print("Rekomendasi untuk", kdrama_title)
recommend_kdrama_weighted(kdrama_title)
```

![Screenshot 2025-05-08 130505](https://github.com/user-attachments/assets/aae3b2a8-c7d0-404c-8378-d4ad2b7780ec)


Metode ini memiliki beberapa kelebihan antara lain memperhitungkan preferensi implisit dari Rating dan memberikan bobot lebih pada K-Drama yang disukai. Di sisi lain juga terdapat kekurangan yaitu berkemungkinan bias terhadap K-Drama dengan Rating tinggi tetapi kontennya kurang relevan secara keseluruhan.

## Evaluation

Pada tahap ini, sistem dievaluasi secara kualitatif dan berbasis konten. Evaluasi dilakukan dengan cara:
1. Mengukur rata-rata skor similarity antara film acuan dan hasil rekomendasi.
2. Melibatkan observasi manual terhadap kecocokan antara Genre dan Rating.

Pertama-tama akan diambil sebuah baris data K-Drama secara acak.
```
# Ambil satu baris acak dari DataFrame
random_row = df.sample(1).iloc[0]

# Ambil informasi Title dan Genre
random_title = random_row['Title']
random_genre = random_row['Genre']

print("Memilih K-Drama acak:", random_title)
print("Genre:", random_genre)
```

![Screenshot 2025-05-08 131633](https://github.com/user-attachments/assets/4c1ecba6-29a6-4833-a285-e8a2a8269615)

Kemudian, sistem memberikan rekomendasi berdasarkan data tersebut.
```
recommend_kdrama_tfidf(random_title)
```

![Screenshot 2025-05-08 131513](https://github.com/user-attachments/assets/cb57dafb-93f2-47a4-9718-88a1941d2d81)

```
recommend_kdrama_weighted(random_title)
```

![Screenshot 2025-05-08 131531](https://github.com/user-attachments/assets/3d13793e-7e68-4c6a-807e-0f4e06bdbad6)


Dari salah satu K-Drama yang dipilih secara acak, sistem menghasilkan rekomendasi dengan nilai similarity score rata-rata sebesar 0.7444 (untuk metode TF-IDF pada Genre) dan 0.7732 (untuk metode Genre Embedding dengan Pembobotan Rating). Nilai ini menunjukkan bahwa K-Drama yang direkomendasikan memiliki tingkat kemiripan konten yang tinggi terhadap K-Drama acuan, berdasarkan representasi Genre dan Rating. Selain similarity score, evaluasi dengan observasi manual dilakukan dengan mencocokkan Genre dan Rating antara K-Drama acuan dan hasil rekomendasi. Genre utama dari film acuan (misalnya "Thriller, Drama") juga muncul pada sebagian besar hasil rekomendasi. Ini menunjukkan bahwa sistem berhasil mengenali atribut yang menjadi ciri khas K-Drama dan merekomendasikan K-Drama lain dengan karakteristik serupa. Rating K-Drama yang direkomendasikan juga cenderung berada dalam kisaran yang sama atau lebih tinggi dari K-Drama acuan, menandakan bahwa sistem tidak menurunkan standar kualitas rekomendasi berdasarkan Rating.

Sebagai penutup, seluruh rangkaian proyek ini berhasil menjawab masalah bisnis yang dipaparkan sebelumnya. Dengan adanya Solution Statements yang direncanakan (membangun sebuah sistem rekomendasi menggunakan content-based filtering dengan metode TF-IDF pada Genre dan Genre Embedding dengan Pembobotan Rating), maka berdasarkan hasil evaluasi, masalah dan tujuan dapat terjawab dimana solusi dinilai mampu memberikan K-Drama sesuai preferensi pengguna sekaligus menguntungkan juga bagi penyedia layanan digital untuk membujuk pengguna agar betah menggunakan jasa layanannya.
