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
Tahapan data preparation merupakan proses penting dalam membangun model machine learning, termasuk sistem rekomendasi. Pada proyek ini, Data Preparation dilakukan untuk memastikan bahwa data bersih, relevan, dan dalam format yang sesuai untuk digunakan dalam pemodelan content-based filtering. Sebelum dilakukan pembersihan, dataset awal kdramas dilakukan seleksi atribut untuk memilah atribut yang hanya akan digunakan untuk modeling yaitu Title, Rating dan Genre.
```
required_attributes = ['Title', 'Rating', 'Genre']
df = kdramas[required_attributes]
```

### Penanganan Missing Values
Kemudian, dari dataset tersebut dilanjutkan dengan tahap pengecekan apakah ada missing values pada data. Hal ini dilakukan untuk memastikan data bersih sehingga tidak bias dalam tahap pemodelan nantinya.
```
print(df.isnull().sum())
```
![Screenshot 2025-05-13 100058](https://github.com/user-attachments/assets/b44da660-dc9e-4256-bc19-48121b1aa58a)

Data terlihat tidak ada missing values sehingga tahap ini dapat diabaikan.

### Penanganan Data Duplikat
Salah satu langkah penting lainnya adalah menangani data duplikat. Data duplikat adalah baris-baris yang memiliki nilai identik pada satu atau lebih kolom dalam dataset yang dapat menyebabkan ketidaktepatan dalam modeling.
```
duplicate_count = df.duplicated().sum()
print(f"Jumlah duplikat: {duplicate_count}")
```
![Screenshot 2025-05-13 100736](https://github.com/user-attachments/assets/2051c4e9-4eb3-4d58-bac3-89aff3977b20)

Dari hasil eksekusi kode tersebut, dapat terlihat bahwa tidak adanya data duplikat pada dataset sehingga tahap ini juga dapat diabaikan.

### Standarisasi Atribut Genre
Pada dataset, Genre berisi nilai dalam bentuk string yang mencantumkan beberapa genre film, misalnya "Action, Comedy, Drama". Untuk memudahkan analisis dan pemrosesan data, pada proyek ini dilakukan standarisasi Genre agar nilai dalam kolom tersebut berada dalam format yang lebih konsisten dan mudah dikelola, seperti list yang berisi genre-genre dalam huruf kecil.
```
df.loc[:, 'GenreList'] = df.loc[:, 'Genre'].apply(lambda x: [genre.strip().lower() for genre in x.split(',')])
```
![Screenshot 2025-05-13 101716](https://github.com/user-attachments/assets/de46d243-cbfa-4d96-86ce-b2760e9cf4f1)


### Normalisasi Atribut Rating
Dalam proyek ini, atribut numerik Rating dalam dataset perlu dinormalisasi atau diskalakan agar berada dalam rentang tertentu. Selain untuk memudahkan analisis dan pemrosesan data, tahap ini juga tidak kalah penting untuk pembangunan model yang sensitif terhadap skala data. Misalkan dataset memiliki Rating yang berupa angka desimal, seperti 9.2, 9.1, 9.0, dan seterusnya. Nilai-nilai ini perlu dinormalisasi agar berada dalam rentang yang seragam, seperti [0 - 1].
```
scaler = MinMaxScaler()
df.loc[:, 'RatingNorm'] = scaler.fit_transform(df[['Rating']])
```
![Screenshot 2025-05-13 103535](https://github.com/user-attachments/assets/a7a518d4-3e22-462d-9ff3-8f88583e5a84)


## Modeling

### TF-IDF pada Genre

Metode ini menggunakan TF-IDF Vectorization terhadap fitur Genre K-Drama. Setiap K-Drama diubah menjadi representasi vektor berdasarkan Genre-nya. Kemudian, digunakan cosine similarity untuk mengukur kemiripan antar K-Drama. Adapun tahapan yang dilakukan sebagai berikut:
1. Melakukan vectorization dengan TfidfVectorizer.
2. Menghitung similarity antar K-Drama menggunakan cosine_similarity.
3. Mengambil Top-N K-Drama paling mirip sebagai rekomendasi.

```
# Mengambil salah satu Title acak dari dataframe
kdrama_title = df['Title'].sample(1).iloc[0]
print("Title:", kdrama_title)
print("GenreList:", df[df['Title'] == kdrama_title]['GenreList'].values[0])
recommend_kdrama_tfidf(kdrama_title)
```

![Screenshot 2025-05-13 112541](https://github.com/user-attachments/assets/c93eadf4-5007-4e34-a257-03def687ad4e)

Metode ini memiliki beberapa kelebihan antara lain data interaksi pengguna tidak diperlukan dan hasil rekomendasi dapat dijelaskan dengan fitur konten (Genre). Di sisi lain juga terdapat beberapa kekurangan yaitu terbatas pada informasi yang ada dalam K-Drama itu sendiri yang berarti terdapat masalah cold-start untuk K-Drama baru dengan Genre unik.

### Genre Embedding dengan Pembobotan Rating

Metode ini menggunakan pendekatan serupa dengan metode yang sebelumnya, namun dengan pembobotan berdasarkan Rating. Genre diubah menjadi representasi biner/embedding, kemudian dilakukan pembobotan berdasarkan rating, agar Genre dari K-Drama dengan Rating tinggi mendapatkan bobot lebih besar dalam menentukan kemiripan. Adapun tahapan yang dilakukan sebagai berikut:
1. Mengubah fitur Genre ke dalam bentuk multi-hot encoding.
2. Melakukan pembobotan berdasarkan rating.
3. Menghitung cosine similarity antar vektor yang telah dibobot.
4. Menghasilkan rekomendasi berdasarkan skor kemiripan tertinggi.

```
print("Title:", kdrama_title)
print("GenreList:", df[df['Title'] == kdrama_title]['GenreList'].values[0])
recommend_kdrama_weighted(kdrama_title)
```

![Screenshot 2025-05-13 113200](https://github.com/user-attachments/assets/0b0ea23d-35a8-48e6-b00d-da1fc0410a71)

Metode ini memiliki beberapa kelebihan antara lain memperhitungkan preferensi implisit dari Rating dan memberikan bobot lebih pada K-Drama yang disukai. Di sisi lain juga terdapat kekurangan yaitu berkemungkinan bias terhadap K-Drama dengan Rating tinggi tetapi kontennya kurang relevan secara keseluruhan.


## Evaluation

Tahap ini bertujuan untuk mengukur kinerja sistem rekomendasi dalam merekomendasikan judul-judul K-Drama yang relevan bagi pengguna. Adapun metrik-metrik yang digunakan yaitu Precision, Recall, dan F1 Score. Metrik tersebut sangat cocok karena sistem rekomendasi content-based filtering seringkali bekerja dengan keterbatasan umpan balik eksplisit dari pengguna. Sebelum penjelasan metrik, terdapat beberapa variabel yang perlu diketahui:

| Variabel | Keterangan |
|----------|----------|
| `TP (True Positives)` | Judul yang direkomendasikan dan juga disukai pengguna |
| `FP (False Positives)` | Judul yang direkomendasikan tetapi tidak disukai pengguna |
| `FP (False Positives)` | Judul yang disukai pengguna tetapi tidak direkomendasikan |

### Precision
Precision mengukur proporsi rekomendasi yang relevan dari seluruh rekomendasi yang diberikan. Dengan kata lain, seberapa akurat sistem dalam memberikan hasil yang benar.

![Screenshot 2025-05-13 131100](https://github.com/user-attachments/assets/4de218af-4097-42ca-94e0-bde30ea23649)


### Recall
Recall mengukur seberapa banyak dari semua judul yang disukai pengguna berhasil ditemukan oleh sistem rekomendasi.

![Screenshot 2025-05-13 131109](https://github.com/user-attachments/assets/d0a98792-5154-4e81-af78-8a9491ddcdf7)


### F1 Score
F1 Score adalah rata-rata harmonik dari Precision dan Recall. Metrik ini digunakan untuk menyeimbangkan antara kemampuan sistem dalam memberikan rekomendasi yang akurat dan lengkap.

![Screenshot 2025-05-13 131120](https://github.com/user-attachments/assets/bb5d6a3c-cb81-4dbf-b52f-36038dd2b542)


### Hasil Evaluasi

Karena tidak tersedia data eksplisit atau feedback pengguna, maka evaluasi dilakukan menggunakan asumsi daftar judul yang disukai oleh pengguna.
```
user_likes = ["Moving", "Memorist", "The Uncanny Counter", "Vampire Prosecutor 2"]
```

Daftar user_likes digunakan untuk menilai kualitas rekomendasi. Sistem dianggap memberikan rekomendasi yang relevan jika hasilnya juga termasuk dalam daftar user_likes pengguna (kecuali judul input itu sendiri). Evaluasi dilakukan dengan mengukur Precision, Recall, dan F1 Score untuk setiap judul yang disukai pengguna, lalu dihitung rata-ratanya. Berikut adalah hasil evaluasi.

![Screenshot 2025-05-13 134117](https://github.com/user-attachments/assets/e080c304-c10e-4a38-b59e-a1c007beebf8)

- Kedua model berhasil mencapai Precision 1.0, yang berarti semua rekomendasi yang diberikan benar-benar relevan. Ini menunjukkan bahwa kedua sistem tidak memberikan rekomendasi yang keliru.
- Recall dari model TF-IDF (0.50) lebih tinggi dibandingkan model Weighted (0.30), yang berarti model TF-IDF berhasil menemukan lebih banyak judul lain yang memang disukai pengguna sedangkan model Weighted melewatkan lebih banyak judul yang seharusnya direkomendasikan.
- F1 Score TF-IDF (0.6607) lebih tinggi daripada Weighted (0.4524), menunjukkan bahwa secara keseluruhan model TF-IDF lebih seimbang dalam hal akurasi dan cakupan sedangkan model Weighted meskipun akurat, kurang luas dalam menjangkau preferensi pengguna.

Sebagai penutup, seluruh rangkaian proyek ini berhasil menjawab masalah bisnis yang dipaparkan sebelumnya. Dengan adanya Solution Statements yang direncanakan (membangun sebuah sistem rekomendasi menggunakan content-based filtering dengan metode TF-IDF pada Genre dan Genre Embedding dengan Pembobotan Rating), maka berdasarkan hasil evaluasi, masalah dan tujuan dapat terjawab dimana solusi dinilai mampu memberikan K-Drama sesuai preferensi pengguna sekaligus menguntungkan juga bagi penyedia layanan digital untuk membujuk pengguna agar betah menggunakan jasa layanannya.
