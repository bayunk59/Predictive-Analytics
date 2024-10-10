# Laporan Proyek Machine Learning - Bayun Kurniawan

## Domain Proyek

Cuaca menjadi salah satu pengaruh terbesar dalam aspek kehidupan makhluk hidup terutama manusia, mulai dari melakukan perencanaan aktivitas sehari-hari hingga pengambilan keputusan dalam sektor seperti pertanian, penerbangan, transportasi dan pariwisata. Akurasi dalam mempredikasi cuaca sangat penting untuk mengurangi dampak buruk dari cuaca, seperti banjir, kekeringan atau bahkan badai. Selain itu, banyak sektor bisnis yang memerlukan data cuaca untuk memperkirakan permintaan produk konsumen.

Dengan menggunakan predictive analytics pada machine learning, diharapkan dapat menghasilkan permodelan cuaca yang bagus dan dapat digunakan dikemudian hari. Melalui predicitive analytics kita dapat menggunakan data cuaca beserta fiturnya untuk mengidentifikasi polanya dan melakukan prediksi jenis cuaca dengan beberapa faktor.

## Business Understanding

Bagian laporan ini mencakup:

### Problem Statements

Saat ini prediksi cuaca masih menjadi komoditas yang cukup menarik untuk didapatkan karena hasil prediksinya dapat digunakan di banyak sektor bisnis. Oleh karena hal tersebut sangat penting dalam menentukan:

1. Faktor apa saja yang berpengaruh pada tipe cuaca?
2. Bagaimana memprediksi tipe cuaca secara akurat berdasarkan variabel-variabel tersebut?

### Goals

Proyek ini memiliki beberapa tujuan utama yang ingin dicapai, antara lain:

1. Mengidentifikasi faktor-faktor utama yang mempengaruhi tipe cuaca
2. Membangun model prediksi cuaca yang akurat dengan mengembangkan model machine learning untuk memprediksi tipe cuaca berdasarkan data historis dengan menggunakan teknik klasifikasi yang tepat

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

Jumlah data adalah 13200 dengan jumlah outlier 1806 atau 13,6% dari data. Data ini digunakan untuk menentukan tipe cuaca berdasarkan beberapa fitur di atas.

**Rubrik/Kriteria Tambahan (Opsional)**:

Tahapan yang akan saya lakukan adalah:

- Data loading :
  saya mengambil data tersebut dari kaggle

- Exploratory Data Analysis - Deskripsi Variabel :
  Data tersebut berisi 13.200 data dengan 11 fitur dimana 7 numeric dan 4 categorical. Dari data tersebut tidak ditemukan data null tetapi banyak terdapat outlier sesuai dengan penjelasan di kaggle

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
  Saya akan menggunakan teknik StandarScaler dari library Scikitlearn, StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

## Modeling

Pada tahap ini, saya akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, saya akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan saya gunakan, antara lain:

1. K-Nearest Neighbor
2. Random Forest
3. Boosting Algorithm

dengan tahapan sebagai berikut:

- Pertama saya siapkan data yang akan digunakan untuk model
- Selanjutnya saya melatih dengan K-Nearest Neighbor terlebih dahulu, nilai k yang saya gunakan adalah 10 tetangga dan menggunakan metric Euclidean. nilai k ini menunjukkan titik data terdekat (tetangga), sedangkan metric euclidean adalah cara paling umum untuk mengukur jarak antar dua ruang mutidimensi
- Model ini mudah dipahami dan digunakan tetapi selalu ada kekurangannya. kekurangannya muncul jika jumlah fiturnya sangat banyak
- Selanjutnya kita akan melatihnya menggunakan Random Forest dengan bantuan library scikit-learn
- Beberapa parameter yang digunakan adalah n_estimator (jumlah pohon), max_depth (kedalaman /panjang pohon), random_state (random number), n_jobs (jumlah pekerjaan yang dilakukan secara paralel)
  -Yang terakhir, saya akan menggunakan model Boosting Algorithm dengan metode adaptive boosting. Dalam kasus ini awalnya semua data latih memiliki weight atau bobot yang sama, kemudian model diperiksa apakah observasi yang dilakukan sudah benar, jika benar bobot akan lebih tinggi yang kemudian akan diberikan pada bobot yang kecil sehinggan akan masuk ke tahap berikutnya. Prosesnya akan berulang terus sampai model mencapau akurasi yang diinginkan
- Parameter yang digunakan dalam model ini adalah learning_rate (bobot yang ditetapkan pada setiap regresor) dan random_statr (digunakan untuk mengontrol random number generator yang digunakan)

## Evaluation

**akurasi, precision, recall, dan F1 score**.

- Metrik yang saya gunakan adalah MSE atau Mas Square Error yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi.
- 1. Saya melakukan scalling antara data uji dan data latih
  2. Selanjutnya di evaluasi menggunakan metriks MSE
  3. Berdasarkan evaluasinya, didapatkan nilai

  <picture>
   <img alt="evaluasi" src="https://github.com/bayunk59/Predictive-Analytics/blob/8eb482327e481fc83ff1fc07a5a6c7b403bc54f8/Cuplikan%20layar%202024-10-08%20213620.png">
  </picture>

  4. kita plot metrik nilai tersebut dengan bar chart

  <picture>
   <img alt="plot" src="https://github.com/bayunk59/Predictive-Analytics/blob/3084490391ff56206ed036d9be5678ce7ebc4002/Cuplikan%20layar%202024-10-08%20214355.png">
  </picture>

  5. Cek Akurasinya
     <picture>
     <img alt="akurasi" src="https://github.com/bayunk59/Predictive-Analytics/blob/3084490391ff56206ed036d9be5678ce7ebc4002/Cuplikan%20layar%202024-10-08%20214416.png">
     </picture>

  pada akurasi tersebut hasilnya menunjukkan model Random Forest memiliki nilai 88,71%, K-Nearest Neighbor memiliki nilai 84,47% sedangkan model Boosting Algorithm memiliki nilai 64,24. Saat di cek prediksinya

  6. Cek prediksi
     <picture>
     <img alt="model" src="https://github.com/bayunk59/Predictive-Analytics/blob/3084490391ff56206ed036d9be5678ce7ebc4002/Cuplikan%20layar%202024-10-08%20214434.png">
     </picture>

  Random Forest menunjukkan nilai 1 yang sesuai dengan nilai sebenarnya, sedangkan K-Nearest Neighbor dan Boosting Algorithm sama-sama bernilai 0,9

  Berdasarkan evaluasi dan modellingnya didapatkan kesimpulan:

  1. Fitur yang mempunyai korelasi tertinggi dengaan Weather Type adalah fitur UV Index
  2. Model terbaik yang muncul adalah Random Forest dengan nilai akurasi sebesar 88,64%

**---Ini adalah bagian akhir laporan---**

_Catatan:_

- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.
