# Laporan Proyek Machine Learning - Klasifikasi Cuaca

## Domain Proyek

Cuaca menjadi salah satu pengaruh terbesar dalam aspek kehidupan makhluk hidup terutama manusia, mulai dari melakukan perencanaan aktivitas sehari-hari hingga pengambilan keputusan dalam sektor seperti pertanian, penerbangan, transportasi dan pariwisata. Cuaca yang tak menentu dapat membuat jadwal yang dibuat menjadi berantakan, salah satu solusi yang bisa dilakukan adalah mengklasifikasikan cuaca. Dengan mengklasifikasikan cuaca dapat dilihat anomali apa saja yang terjadi dan hasilnya dapat dipelajar untuk kebutuhan riset di kemudian hari.

Klasifikasi cuaca adalah proses mengelompokkan cuaca berdasarkan karakteristik atau pola tertentu yang telah ada dalam data. Klasifikasi cuaca biasa digunakan untuk mengelompokkan data cuaca berdasarkan beberapa kondisi atau pola yang ada, ini bertujuan untuk menghasilkan informasi cuaca yang lebih mudah dipahami dan bisa digunakan dalam berbagai konteks.

Berbeda dengan prediksi cuaca, klasifikasi cuaca bisa dikatakan lebih pasti karena didasarkan pada data yang sudah pernah terjadi. Hasil klasifikasi cuaca biasanya digunakan dalam riset iklim, analisis tren cuaca, serta pembuatan aplikasi yang membutuhkan pengenalan pola cuaca.

Untuk melakukan klasifikasi cuaca dengan baik, dapat digunakan permodelan machine learning. Machine learning bisa mengklasifikasikan cuaca tersebut berdasarkan fitur-fitur yang sudah ada. Dalam machine learning sendiri terdapat banyak sekali algoritma untuk membuat permodelan klasifikasi. Beberapa permodelan yang bisa dipakai adalah K-Nearest Neighbor (KNN), Random Forest dan Boosting Algoritm. Dengan memanfaatkan berbagai data yang didapat dan permodelan machine learning diharapkan klasifikasi cuaca ini mendapatkan hasil yang diinginkan dan dapat bermanfaat di kemudian hari.

## Business Understanding

Berdasarkan data yang diambil dari kaggle mengenai klasifikasi cuaca, terdapat banyak faktor yang yang mempegaruhi sebuah perubahan cuaca. Maka dari itu, dibutuhkannya pengembangan model machine learning untuk membantu dan menentukan faktor apa saja yang berpengaruh pada tipe-tipe cuaca tertentu.

### Problem Statements

Problem statements yang ingin kami bahas adalah:

1. Berdasarkan dataset yang kami gunakan, fitur-fitur apa saja yang membedakan tipe cuaca yang satu dengan yang lainnya?
2. Bagaimana cara mendapatkan model terbaik untuk klasifikasi cuaca tersebut?

### Goals

Goals/tujuan dari poyek ini adalah:

1. Melakukan eksplorasi pada semua fitur untuk menentukan fitur mana saja yang memiliki pengaruh besar atau korelasi tertinggi dengan tipe cuaca tersebut.
2. Melakukan proses training terhadap beberapa model yang kami gunakan di proyek ini.

### Solution statements

Beberapa solusi yang akan coba terapkan adalah:

1. Melakukan eksplorasi fitur manggunakan analisis univariat dan multivariat untuk menemukan hubungan antar fitur baik yang data numerik maupun data kategorikal.
2. Untuk mendapatkan data yang bersih sebelum di buat permodelan. Dilakukan preparation data yang terdiri dari Encoding Fitur Kategori, Train-Test-Spit dan Standarisasi.
3. Permodelan akan dilakukan dengan 3 algoritma model, yaitu `K-Nearest Neighbors (KNN)`, `Random Forest (RF)` dan `Boosting Algorithm` lalu akan dipilih model terbaik berdasarkan nilai akurasinya.

## Data Understanding

Data yang saya gunakan berasal dari Kaggle dengan judul Weather Type Classification [cuaca](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification), Data tersebut berisi 13200 data dan 11 kolom dengan 7 data numerik dan 4 data katgorik. Berikut adalah detail masing-masing kolomnya:

### Variabel-variabel pada dataset adalah sebagai berikut:

- `Temperature` (numeric) : Temperatur suhu dalam celcius
- `Humidity` (numeric) : Presentase Kelembaban
- `Wind Speed` (numeric) : Kecepatan angin dalam kilometer/jam
- `Precipitation (%)` (numeric) : Presentase curah hujan
- `Cloud Cover` (categorical) : Deskripsi tutupan awan yang berisi clear, cloudy, overcast dan party cloudy
- `Atmospheric Pressure` (numeric) : Tekanan atmosfer dalam hPa
- `UV index` (numeric) : Indeks UX yang menunjukkan kekuatan radiasi UV
- `Season` (categorical) : Jenis musim mulai dari Autumn, Spring, Summer dan Winter
- `Visibility (km)` (numeric) : Jarak pandang dalam km
- `Location` (categorical) : Lokasi dimana data di ambil seperti coastal, inland dan muntain
- `Weather Type` (categorical) : Jenis cuaca yang berisi Cloudy, Rainy, Snowy dan Sunny (Target Klasifikasi)

Tahapan yang akan saya lakukan adalah:

### Exploratory Data Analysis

Exploratory data analysis atau sering disingkat EDA merupakan proses investigasi awal pada data untuk menganalisis karakteristik, menemukan pola, anomali, dan memeriksa asumsi pada data. Teknik ini biasanya menggunakan bantuan statistik dan representasi grafis atau visualisasi.

Berikut adalah tahapan EDA yang dilakukan

1. cek nilai duplikat pada data

```
duplicate_rows = cuaca[cuaca.duplicated()]
print("Jumlah baris duplikat:", duplicate_rows.shape[0])
```

Output: Tidak terdapat baris duplikat

2. cek nilai yang kosong pada data

```
print(cuaca.isnull().sum())
```

output:

```
Temperature             0
Humidity                0
Wind Speed              0
Precipitation (%)       0
Cloud Cover             0
Atmospheric Pressure    0
UV Index                0
Season                  0
Visibility (km)         0
Location                0
Weather Type            0
dtype: int64
```

3. Mengubah type data target, dalam ini `Weather Type` dari kategori menjadi numerik agar mudah dalam permodelan

```
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
cuaca['Weather Type'] = le.fit_transform(cuaca['Weather Type'])
cuaca.head()
```

output:

```
| Temperature | Humidity | Wind Speed | Precipitation (%) | Cloud Cover   | Atmospheric Pressure | UV Index | Season | Visibility (km) | Location | Weather Type |
|-------------|----------|------------|-------------------|---------------|----------------------|----------|--------|-----------------|----------|--------------|
| 14.0        | 73       | 9.5        | 82.0              | partly cloudy | 1010.82              | 2        | Winter | 3.5             | inland   | 1            |
| 39.0        | 96       | 8.5        | 71.0              | partly cloudy | 1011.43              | 7        | Spring | 10.0            | inland   | 0            |
| 30.0        | 64       | 7.0        | 16.0              | clear         | 1018.72              | 5        | Spring | 5.5             | mountain | 3            |
| 38.0        | 83       | 1.5        | 82.0              | clear         | 1026.25              | 7        | Spring | 1.0             | coastal  | 3            |
| 27.0        | 74       | 17.0       | 66.0              | overcast      | 990.67               | 1        | Winter | 2.5             | mountain | 1            |

```
Ubah tipe data berhasil, tipe data `Weather Type` berubah menjadi numerik dengan rincian:
0 = Cloudy
1 = Rainy
2 = Snowy
3 = Sunny

4. Menangani Outlier
   Outlier adalah titik data yang secara signifikan berada di sebgaian data dalam kumpulan data. Outlier ini bisa muncul karena banyak faktor salah satunya adalah kesalahan pengamatan.
   - Menampilkan data outlier
     ```
     for column in cuaca.select_dtypes(include=np.number).columns:
     plt.figure(figsize=(8, 6))
     sns.boxplot(x=cuaca[column])
     plt.title(f'Boxplot of {column}')
     plt.show()
     ```
     output:
     ![outlier1](https://github.com/user-attachments/assets/233ce168-9edd-4823-b346-9c895d9435b4)
     ![outlier3](https://github.com/user-attachments/assets/1e0c1417-06d9-47eb-bc20-9cb80bc10512)
     ![outlier5](https://github.com/user-attachments/assets/f7d3f274-a220-45d8-80ba-39b330c48cdd)
     ![outlier7](https://github.com/user-attachments/assets/6bb6db3f-88aa-4cbd-a4e3-85db348acab0)
     
     berdasarkan boxplot tersebut, ada 4 fitur yang memiliki outlier yakni fitur `Temperature`, `Wind           Speed`, `Athmospheric Pressure`, dan `Visibility (km)`
   - Outlier pada 4 fitur tersebut perlu dihapus untuk mendapatkan model yang bagus
     ```
     numeric_cuaca = cuaca.select_dtypes(include=np.number)

     Q1 = numeric_cuaca.quantile(0.25)
     Q3 = numeric_cuaca.quantile(0.75)
     IQR = Q3 - Q1
     lower_bound = Q1 - 1.5 * IQR
     upper_bound = Q3 + 1.5 * IQR

     cuaca = cuaca[~((numeric_cuaca < lower_bound) | (numeric_cuaca > upper_bound)).any(axis=1)]
     ```
     Outlier telah berhasil dihapus.
   - Menampilkan data terbaru setelah outliers dihapus
     ```
     cuaca.shape
     ```
     output:
     ``
     (11689, 11)
     ``

     Jumlah data terbaru sekarang adalah 11689 data

## Data Preparation

Pada bagian ini kami akan melakukan tiga tahap persiapan data, yaitu:

1. Encoding fitur kategori.
   Data yang akan saya ubah adalah fitur kategori yaitu 'Cloud Cover', 'Location' dan 'Season'. Proses ini dilakukan dengan menggunakan teknik one-hot-encoding agar didapatkan fitur baru yang sesuai sehingga dapat mewakili fitur kategori

2. Pembagian dataset dengan fungsi train_test_split.
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
- Selanjutnya kami akan melatihnya menggunakan Random Forest dengan bantuan library scikit-learn
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

  4. kami plot metrik nilai tersebut dengan bar chart

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
