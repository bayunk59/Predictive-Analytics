# Laporan Proyek Machine Learning - Bayun Kurniawan

## Domain Proyek

Cuaca menjadi salah satu pengaruh terbesar dalam aspek kehidupan makhluk hidup terutama manusia, mulai dari melakukan perencanaan aktivitas sehari-hari hingga pengambilan keputusan dalam sektor seperti pertanian, penerbangan, transportasi dan pariwisata. Akurasi dalam mempredikasi cuaca sangat penting untuk mengurangi dampak buruk dari cuaca, seperti banjir, kekeringan atau bahkan badai. Selain itu, banyak sektor bisnis yang memerlukan data cuaca untuk memperkirakan permintaan produk konsumen.

Dengan menggunakan predictive analytics pada machine learning, diharapkan dapat menghasilkan permodelan cuaca yang bagus dan dapat digunakan dikemudian hari. Melalui predicitive analytics kita dapat menggunakan data cuaca beserta fiturnya untuk mengidentifikasi polanya dan melakukan prediksi jenis cuaca dengan beberapa faktor.

**Rubrik/Kriteria Tambahan (Opsional)**:

## Business Understanding

Bagian laporan ini mencakup:

### Problem Statements

Saat ini prediksi cuaca masih menjadi komoditas yang cukup menarik untuk didapatkan karena hasil prediksinya dapat digunakan di banyak sektor bisnis. Oleh karena hal tersebut sangat penting dalam menentukan:

1. Faktor apa saja yang berpengaruh pada tipe cuaca?
2. Bagaimana memprediksi tipe cuaca secara akurat berdasarkan variabel-variabel tersebut?

### Goals

Proyek ini memiliki beberapa tujuan utama yang ingin dicapai, antara lain:

1. Mengidentifikasi faktor-faktor utama yang mempengaruhi tipe cuaca
   Menentukan variabel cuaca mana yang paling berpengaruh terhadap jenis cuaca
2. Membangun model prediksi cuaca yang akurat dengan mengembangkan model machine learning untuk memprediksi tipe cuaca berdasarkan data historis dengan menggunakan teknik klasifikasi yang tepat

**Rubrik/Kriteria Tambahan (Opsional)**:

- Menambahkan bagian “Solution Statement” yang menguraikan cara untuk meraih goals. Bagian ini dibuat dengan ketentuan sebagai berikut:

  ### Solution statements

  - Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
  - Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding

### Variabel-variabel pada dataset adalah sebagai berikut:

Data yang saya gunakan berasal dari Kaggle dengan judul Weather Type Classification (https://www.kaggle.com/datasets/nikhil7280/weather-type-classification), yang berisi beberapa fitur data sebagai berikut:

- Temperature (numeric) : Temperatur suhu dalam celcius
- Humidity (numeric) : Presentase Kelembaban
- Wind Speed (numeric) : Kecepatan angin dalam kilometer/jam
- Precipitation (%) (numeric) : Presentase curah hujan
- Cloud Cover (categorical) : Deskripsi tutupan awan yang berisi clear, cloudy, overcast dan party cloudy
- Atmospheric Pressure (numeric) : Tekanan atmosfer dalam hPa
- UV index (numeric) : Indeks UX yang menunjukkan kekuatan radiasi UV
- Season (categorical) : Jenis musim mulai dari Autumn, Spring, Summer dan Winter
- Visibility (km) (numeric) : Jarak pandang dalam km
- Location (categorical) : Lokasi dimana data di ambil seperti coastal, inland dan muntain
- Weather Type (categorical) : Jenis cuaca yang berisi Cloudy, Rainy, Snowy dan Sunny (Target Klasifikasi)

**Rubrik/Kriteria Tambahan (Opsional)**:

Tahapan yang akan saya lakukan adalah:

- Data loading :
  saya akan mengambil data tersebut dari kaggle

- Exploratory Data Analysis - Deskripsi Variabel :
  Data tersebut berisi 13.200 data dengan 11 fitur dimana 7 numeric dan 4 categorical.

- Exploratory Data Analysis - Menangani data:

  1. Saya akan merubah tipe data fitur target (Weather Type) dari categorical menjadi numerik agar mempermudah prediksi
     cloudy: 0, Rainy: 1, Snowy: 2 dan Sunny: 3
  2. Saya akan mengecek jumlah outlier di data numerik lalu menghapusnya.
     beberapa fitur yang memiliki outlier adalah Temperatur 92 outlier, Wind Speed 404 outlier, Atmospheric Pressure 927 outlier dan Visibility (km) 383 outlier.
     Karena outliernya banyak, daripada menghapusnya saya akan coba menanganinya dengan cara menggantinya dengan nilai median

- Exploratory Data Analysis - Univariate Analysis:

  1. Saya membagi datanya menjadi 2 fitur (numeric dan categorical)
  2. di fitur categorical dan numeric, saya melihat sebaran data di tiap fiturnya

- Exploratory Data Analysis - Multivariate Analysis:
  1. pada categorical fitur, saya melihat hubungan masing-masing fitur dengan target Weather Type
  2. saya mencoba mengubah categorical fitur menjadi numerik untuk melihat korelasinya
  3. Pada numerical fitur, saya menggunakan fungsi pairplot() dan saya juga akan mengobservasi korelasi antara fitur numerik dengan fitur target

## Data Preparation

Pada bagian ini kita akan melakukan tiga tahap persiapan data, yaitu:

- Encoding fitur kategori.
  Data yang akan saya ubah adalah fitur 'Cloud Cover', 'Location' dan 'Season'

- Pembagian dataset dengan fungsi train_test_split dari library sklearn.
  Saya akan menggunakan proporsi pembagian sebesar 90:10

- Standarisasi.

**Rubrik/Kriteria Tambahan (Opsional)**:

## Modeling

Pada tahap ini, saya akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, saya akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan saya gunakan, antara lain:

1. K-Nearest Neighbor
2. Random Forest
3. Boosting Algorithm

   **Rubrik/Kriteria Tambahan (Opsional)**:
   **Jelaskan proses improvement yang dilakukan**.
   **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.
   Berdasarkan hasilnya, Random Forest menjadi pilihan saya karena nilai akurasinya paling tinggi

## Evaluation

**akurasi, precision, recall, dan F1 score**.

- Metrik yang saya gunakan adalah MSE atau Mas Square Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
- 1. Saya melakukan scalling antara data uji dan data latih
  2. Selanjutnya di evaluasi menggunakan metriks MSE
  3. Berdasarkan evaluasinya, didapatkan nilai
     <picture>
     <img alt="evaluasi" src="https://github.com/bayunk59/Predictive-Analytics/blob/8eb482327e481fc83ff1fc07a5a6c7b403bc54f8/Cuplikan%20layar%202024-10-08%20213620.png">
     </picture>

**Rubrik/Kriteria Tambahan (Opsional)**

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
