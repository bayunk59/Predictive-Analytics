# Laporan Proyek Machine Learning - Klasifikasi Cuaca

## Domain Proyek

Cuaca menjadi salah satu pengaruh terbesar dalam aspek kehidupan makhluk hidup terutama manusia, mulai dari melakukan perencanaan aktivitas sehari-hari hingga pengambilan keputusan dalam sektor seperti pertanian, penerbangan, transportasi dan pariwisata. Cuaca yang tak menentu dapat membuat jadwal yang kami buat menjadi berantakan, salah satu solusi yang bisa kami lakukan adalah mengklasifikasikan cuaca. Dengan mengklasifikasikan cuaca kami dapat melihat anomali apa saja yang terjadi dan
kami dapat mempelajarinya untuk kebutuhan riset di kemudian hari.

Klasifikasi cuaca adalah proses mengelompokkan cuaca berdasarkan karakteristik atau pola tertentu yang telah ada dalam data. Klasifikasi cuaca biasa digunakan untuk mengelompokkan data cuaca berdasarkan beberapa kondisi atau pola yang ada, ini bertujuan untuk menghasilkan informasi cuaca yang lebih mudah dipahami dan bisa digunakan dalam berbagai konteks.

Berbeda dengan prediksi cuaca, klasifikasi cuaca bisa dikatakan lebih pasti karena didasarkan pada data yang sudah pernah terjadi. Hasil klasifikasi cuaca biasanya digunakan dalam dalam riset iklim, analisis tren cuaca , sert pembuatan aplikasi yang membutuhkan pengenalan pola cuaca.

Untuk melakukan klasifikasi cuaca dengan baik, kami dapat menggunakan permodelan machine learning. Dengan machine learning kami bisa mengklasifikasikan cuaca tersebut berdasarkan fitur-fitur yang sudah ada. Dalam machine learning sendiri terdapat banyak sekali algoritma untuk membuat permodelan klasifikasi. Beberapa permodelan yang bisa kami pakai adalah K-Nearest Neighbor (KNN), Random Forest dan Boosting Algoritm. Dengan memanfaatkan berbagai data yang didapat dan permodelan machine learning diharapkan klasifikasi cuaca ini mendapatkan hasil yang diinginkan dan dapat bermanfaat di kemudian hari.

## Business Understanding

Berdasarkan data yang kami ambil dari kaggle mengenai klasifikasi cuaca, banyak faktor yang yang mempegaruho sebuah perubahan cuaca. Maka dari itu, dibutuhkannya pengembangan model machine learning untuk membantu dan menentukan faktor apa saja yang berpengaruh pada tipe-tipe cuaca tertentu.

### Problem Statements

Problem statements yang ingin kami bahas adalah:

1. Berdasarkan dataset yang kami gunakan, fitur-fitur apa saja yang membedakan tipe cuaca yang satu dengan yang lainnya?
2. Bagaimana cara mendapatkan model terbaik untuk klasifikasi cuaca tersebut?

### Goals

Goals/tujuan dari poyek ini adalah:

1. Melakukan eksplorasi pada semua fitur untuk menentukan fitur mana saja yang memiliki pengaruh besar atau korelasi tertinggi dengan tipe cuaca tersebut.
2. Melakukan proses training terhadap beberapa model yang kami gunakan di proyek ini.

### Solution statements

Beberapa solusi yang akan kami coba terapkan adalah:

1. Melakukan eksplorasi fitur manggunakan analisis univariat dan multivariat untuk menemukan hubungan antar fitur baik yang data numerik maupun data kategorikal.
2. Untuk mendapatkan data yang berish sebelum di buat permodelan. Dilakukan preparation data yang terdiri dari

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

Tahapan yang akan saya lakukan adalah:

1. Data loading :

- saya mengambil data tersebut dari kaggle Jumlah data adalah 13200 dengan jumlah outlier 1806 atau 13,6% dari data. Data ini digunakan untuk menentukan tipe cuaca berdasarkan beberapa fitur di atas.
- Berdasarkan hasil pengecekan tidak ditemukan duplikat data dan data kosong.

2. Exploratory Data Analysis :

- Data tersebut berisi 13.200 data dengan 11 fitur dimana 7 numeric dan 4 categorical.
- Dari data tersebut terdapat outlier sesuai dengan penjelasan di kaggle
- Saya ubah type data pada target fitur, yaitu Weather Type yang awalnya kategori menjadi numerik agar mudah untuk menemukan korelasi data dengan rincian cloudy: 0, Rainy: 1, Snowy: 2 dan Sunny: 3
- Setelah saya ubah jumlah tipe datanya berubah menjadi 8 numerik dan 3 kategori
- Selanjutnya saya melakukan pengecekan jumlah outlier di data numerik dan ditemukan beberapa fitur yang memiliki outlier adalah Temperatur 92 outlier, Wind Speed 404 outlier, Atmospheric Pressure 927 outlier dan Visibility (km) 383 outlier.
- Karena outliernya banyak, daripada menghapusnya saya akan coba menanganinya dengan cara menggantinya dengan nilai median

3. Exploratory Data Analysis - Univariate Analysis:
   a. Categorical

   - Saya membagi datanya menjadi 2 fitur (numeric dan categorical)
   - di fitur categorical saya menampilkan jumlah datanya dalam bentuk grafik batang
   - terlihat overcast menjadi yang paling banyak dalam fitur cloud cover dengan presentase 46,1%
   - selanjutnya winter pada fitur season memiliki jumlah 5610 dan berbeda dengan 3 season lain yang jumlahnya hampir sama di angka 2500
   - Untuk fitur location, pengamatan paling banyak diambil di lokasi inland dan mountain dengan jumlah 4800 data, sedangkan coastal hanya 3571 data.

     b. numeric

   - Saya menampilkan semua yang memiliki tipe data numerik dalam sebuah histogram
   - Terlihat data beberapa nilai yang sangat menonjol dalam data tersebut seperti pada fitur temperatur, di kisaran angka 20 menjadi data yang terbanyak hingga mencapai 1000 data
   - Pada fitur Humidity juga angak di kisaran 70 menjadi yang terbanyak dengan jumlah hampir 1000 data
   - Pada fitur Wind Speed di kisaran angka 10 menjadi yang terbanyak melebihi 800 data sedangkan yang lain hanya di jumlah 400 data
   - Pada percipitation juga terdapat nilai 60 yang memiliki jumlah paling menonjol dengan jumlah 600 data
   - Pada fitur Atmospheric Pressure juga terlihat jumlah data yang sangat signifikan dengan jumla 1400 data sedangkan yang lain hanya di angka 200 - 700 data
   - Pada UV index dan Visibility juga terdapat data yang menonjol dengan jumlah 2500 dan 1000 data
   - Berbeda dengan data yang lain, jumlah Weather Type nya seimbang
   - Jumlah data yng menonjol ini menurut saya kemungkinan bisa terjadi karena 2 hal:
     1. Karena jumlah data memang sudah seperti itu
     2. Atau bisa jadi karena perubahan data pada outlier menjadi median yang membuat jumlah median bertambah

4. Exploratory Data Analysis - Multivariate Analysis:
   a. Categorical
   - pada categorical fitur, saya melihat hubungan masing-masing fitur dengan target Weather Type
   - Pada fitur Cuaca dan location tidak ada perbedaan data yang signifikan, jadi seharusnya tingkat korelasi kedua fitur tersebut dengan Weather type rendah
   - berbeda dengan fitur cloud cover, ada data clear yang paling berbeda sendiri dan ini menimbulkan kecurigaan saya
   - Akhirnya saya mencoba mengubah categorical fitur menjadi numerik untuk melihat korelasinya
   - dari hasil korelasinya didapatkan nilai -0,54 yang menandakan memang ada korelasi tetapi sifatnya negatif.
   - Karena bernilai negatif, ini menandakan bahwa semakin tinggi tingkat awan (mendung), semakin buruk jenis cuaca yang diprediksi, dan sebaliknya. Hubungan ini berkebalikan dengan data dan akan kami abaikan

b. numeric - Pada numerical fitur, saya menggunakan fungsi pairplot() dan saya juga akan mengobservasi korelasi antara fitur numerik dengan fitur target - Dari hasil korelasi didapatkan bahwa temperatur dan visibility tidak memiliki hubungan dengan target makan akan dihapus untuk mendapatkan permodelan yang lebih baik - Untuk fitur humidity, Wind speed, precipitaion memiliki nilai korelasi negatif - Sedangkan UV Index dan Atmospheric Pressure memiliki nilai korelasi positif dengan nilai 0,35 pada UV Index - Selanjutnya saya membuang fitur Temperature dan Visibility karena tidak ada korelasi dengan target

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
