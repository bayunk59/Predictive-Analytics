# Laporan Proyek Machine Learning - Klasifikasi Cuaca

## Domain Proyek

Cuaca menjadi salah satu pengaruh terbesar dalam aspek kehidupan makhluk hidup terutama manusia, mulai dari melakukan perencanaan aktivitas sehari-hari hingga pengambilan keputusan dalam sektor seperti pertanian, penerbangan, transportasi dan pariwisata. Cuaca yang tak menentu dapat membuat jadwal yang dibuat menjadi berantakan, salah satu solusi yang bisa dilakukan adalah mengklasifikasikan cuaca. Dengan mengklasifikasikan cuaca dapat dilihat anomali apa saja yang terjadi dan hasilnya dapat dipelajar untuk kebutuhan riset di kemudian hari.

Klasifikasi cuaca adalah proses mengelompokkan cuaca berdasarkan karakteristik atau pola tertentu yang telah ada dalam data. Klasifikasi cuaca biasa digunakan untuk mengelompokkan data cuaca berdasarkan beberapa kondisi atau pola yang ada, ini bertujuan untuk menghasilkan informasi cuaca yang lebih mudah dipahami dan bisa digunakan dalam berbagai konteks.

Berbeda dengan prediksi cuaca, klasifikasi cuaca bisa dikatakan lebih pasti karena didasarkan pada data yang sudah pernah terjadi. Hasil klasifikasi cuaca biasanya digunakan dalam riset iklim, analisis tren cuaca, serta pembuatan aplikasi yang membutuhkan pengenalan pola cuaca.

Untuk melakukan klasifikasi cuaca dengan baik, dapat digunakan permodelan machine learning. Machine learning bisa mengklasifikasikan cuaca tersebut berdasarkan fitur-fitur yang sudah ada. Dalam machine learning sendiri terdapat banyak sekali algoritma untuk membuat permodelan klasifikasi. Beberapa permodelan yang bisa dipakai adalah K-Nearest Neighbor (KNN), Random Forest dan Boosting Algoritm. Dengan memanfaatkan berbagai data yang didapat dan permodelan machine learning diharapkan klasifikasi cuaca ini mendapatkan hasil yang diinginkan dan dapat bermanfaat di kemudian hari.

## Business Understanding

Berdasarkan data yang diambil dari kaggle mengenai klasifikasi cuaca, terdapat banyak faktor yang yang mempegaruhi sebuah perubahan cuaca. Maka dari itu, dibutuhkannya pengembangan model machine learning untuk membantu dan menentukan faktor apa saja yang berpengaruh pada tipe-tipe cuaca tertentu.

### Problem Statements

Problem statements yang ingin dibahas adalah:

1. Berdasarkan dataset yang digunakan, fitur-fitur apa saja yang membedakan tipe cuaca yang satu dengan yang lainnya?
2. Bagaimana cara mendapatkan model terbaik untuk klasifikasi cuaca tersebut?

### Goals

Goals/tujuan dari poyek ini adalah:

1. Melakukan eksplorasi pada semua fitur untuk menentukan fitur mana saja yang memiliki pengaruh besar atau korelasi tertinggi dengan tipe cuaca tersebut.
2. Melakukan proses training terhadap beberapa model yang digunakan dalam proyek ini.

### Solution statements

Beberapa solusi yang akan coba terapkan adalah:

1. Melakukan eksplorasi fitur menggunakan analisis univariat dan multivariat untuk menemukan hubungan antar fitur baik yang data numerik maupun data kategorikal.
2. Untuk mendapatkan data yang bersih sebelum di buat permodelan. Dilakukan preparation data yang terdiri dari Menghapus outlier, Menghapus fitur dengan korelasi yang rendah, Encoding Fitur Kategori, Train-Test-Spit dan Standarisasi.
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

| Temperature | Humidity | Wind Speed | Precipitation (%) | Cloud Cover   | Atmospheric Pressure | UV Index | Season | Visibility (km) | Location | Weather Type |
| ----------- | -------- | ---------- | ----------------- | ------------- | -------------------- | -------- | ------ | --------------- | -------- | ------------ |
| 14.0        | 73       | 9.5        | 82.0              | partly cloudy | 1010.82              | 2        | Winter | 3.5             | inland   | 1            |
| 39.0        | 96       | 8.5        | 71.0              | partly cloudy | 1011.43              | 7        | Spring | 10.0            | inland   | 0            |
| 30.0        | 64       | 7.0        | 16.0              | clear         | 1018.72              | 5        | Spring | 5.5             | mountain | 3            |
| 38.0        | 83       | 1.5        | 82.0              | clear         | 1026.25              | 7        | Spring | 1.0             | coastal  | 3            |
| 27.0        | 74       | 17.0       | 66.0              | overcast      | 990.67               | 1        | Winter | 2.5             | mountain | 1            |

Ubah tipe data berhasil, tipe data `Weather Type` berubah menjadi numerik dengan rincian:
0 = Cloudy
1 = Rainy
2 = Snowy
3 = Sunny

### Univariate Analysis

Univariate Analysis adalah jenis analisis data yang memeriksa satu variabel saja. Tujuannya uuntuk menggambarkan data dan menemukan pola distribusi data

Sebelum mulai analysis data akan dibagi menjadi 2 bagian, yakni `numerical_fitur` untuk data numerik dan `categorical_features` untuk data kategorik

```
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)', 'Weather Type']
categorical_features = ['Cloud Cover', 'Season', 'Location']
```

data telah terbagi menjadi numerical_features untuk data numerik dan categorical_features untuk data kategorik

#### Categorical Features

Menampilkan data fitur dalam bentuk grafik

```
feature = categorical_features[2]
count = cuaca[feature].value_counts()
percent = 100*cuaca[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);
```

output:

- Fitur CLoud Cover

  ![uni1](https://github.com/user-attachments/assets/eb54bde9-018c-4812-bd29-512a7e631bd7)

  Berdasarkan grafik pada fitur `Cloud Cover` di atas:

  - `overcast` memiliki 6090 data
  - `party cloud` memiliki 4560 data
  - `clear` memiliki 2139 data
  - `cloudy` memiliki 411 data

- Fitur Season

  ![uni2](https://github.com/user-attachments/assets/89a79189-0ca8-4c7a-aa51-5ff1c5ce3347)

  Berdasarkan grafik pada fitur `Season` di atas:

  - `winter` memiliki 5610 data
  - `Spring` memiliki 2598 data
  - `Autumn` memiliki 2500 data
  - `Summer` memiliki 2492 data
  -

- Fitur Location

  ![uni3](https://github.com/user-attachments/assets/8c64caaf-f523-4d1a-9b46-f8e3d156f937)

  Berdasarkan grafik pada fitur `Location` di atas:

  - `inland` memiliki 4816 data
  - `mountain` memiliki 4813 data
  - `coastal` memiliki 3571 data

- Fitur Weather Type

  ![uni4](https://github.com/user-attachments/assets/54c538c7-c9e4-4724-9716-01aa95a835d5)

  Berdasarkan grafik pada fitur `Weather Type` di atas, nilai pada kolom ini terlihat seimbang dengan rincian sebagai berikut:
- `Rainy` memiliki 3300 data
- `Cluody` memiliki 3300 data
- `Sunny` memiliki 3300 data
- `Snowy` memiliki 3300 data

#### Numerical Features

Menampilkan data numerik dalam bentuk grafik

![uni numerical](https://github.com/user-attachments/assets/86aecae9-6efc-4857-998e-95d1963c7072)

Berdasarkan grafik diatas, hampir semmua kolom skewnessnya mengarah ke kiri kecuali `Humidity`, `Precipitaion (%)` dan `Atmospheric Pressure`.

### Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate Analysis yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate Analysis. Selanjutnya, akan dilakukan analisis data pada fitur kategori dan numerik.

#### Categorical Features

Menampilkan hubungan fitur kategori dengan target `Weather Type`

```
for feature in categorical_features[:-1]:
  plt.figure(figsize=(10, 6))
  sns.countplot(x=feature, hue='Weather Type', data=cuaca)
  plt.title(f'Hubungan {feature} dengan Weather Type')
  plt.xlabel(feature)
  plt.ylabel('Jumlah Data')
  plt.legend(title='Weather Type')
  plt.show()
```

output:

- Fitur `Cloud Cover` dengan `Weather Type`
  ![multi 1](https://github.com/user-attachments/assets/87fc3e98-d64e-40e9-9e40-63b7ac6489c9)

- Fitur `Season` dengan `Weather Type`
  ![multi 2](https://github.com/user-attachments/assets/e5ddf0dc-65e4-4e3d-8855-658535b1614d)

- Fitur `Location` dengan `Weather Type`
  ![multi 3](https://github.com/user-attachments/assets/f6a32fd5-234f-4ae4-93f7-0550e2c9b371)

Berdasarkan grafik di atas didapatn:
-  Pada Fitur `Cloud Cover`
  1. Pada `party cloudy` jumlah `Coudy` menjadi yang terbanyak hampir mendekati 2000 data
  2. pada `Clear` semua nilainya diisi dengan tipe `Sunny`
  3. pada `overccast` nilia terendahnya adda pada `Sunny` dan yang terbanyak adalah `Rainy` dan `Snowy`
  4. pada `cludy`, hampr semua datanya rata

- Pada fitur `Season`
  1. pada `winter` jumlah `Snowy` menjadi yang paling besar melebihi 3000 data, sedangkan yang lain ada di bawah 1000 data
  2. pada `Spring`, `Summer` dan `Autumn` datanya hampir rata antara 500 - 1000 data kecuali pada data `Snowy` yang jumlahnya sangat sedikit

- Pada fitur `Location`
  1. pada `inland` dan `mountain` jumlah datanya hampir sama dengan data tertinggi ada pada `Snowy`
  2. Sedangkan pada `coastal`, data `Snowy` menjdai yang terendah dengan jumlah kurang dari 200 data


#### Numerical Features

Menampilkan hubungan antar fitur numerik dengan target `Weather Type`

```
sns.pairplot(cuaca, hue='Weather Type')
plt.show()
```

output:

![multi numerical](https://github.com/user-attachments/assets/ae924be0-0161-4275-a87b-ef70d05fba11)

Berdasarkan visualisasi data diatas, tidak terlihat adanya hubungan yang signifikan antara fitur dengan target `Weather Type`

## Data Preparation

Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahapan dilakukannya proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan. Dalam data preparation akan dilakukan 3 tahapan, yakni Encoding Fiitur Kategori, Train-Test-Split dan Standarisasi.

### Menangani Outlier

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
  `(11689, 11)`

  Jumlah data terbaru sekarang adalah 11689 data

### Mengubah Type Data

Pada bagian ini, semua data kategorik akan diubah menjadi data numerik untuk mempermudah permodelan dan menentukan nilai uji korelasi antar fitur.

- Mengubah data
  ```
  le = LabelEncoder()
  cuaca['Cloud Cover'] = le.fit_transform(cuaca['Cloud Cover'])
  cuaca['Location'] = le.fit_transform(cuaca['Location'])
  cuaca['Season'] = le.fit_transform(cuaca['Season'])
  cuaca['Weather Type'] = le.fit_transform(cuaca['Weather Type'])
  cuaca.head()
  ```

  |      | Temperature | Humidity | Wind Speed | Precipitation (%) | Cloud Cover | Atmospheric Pressure | UV Index | Season | Visibility (km) | Location | Weather Type |
|------|-------------|----------|------------|-------------------|-------------|----------------------|----------|--------|-----------------|----------|--------------|
| 0    | 14.0        | 73       | 9.5        | 82.0              | 3           | 1010.82              | 2        | 3      | 3.5             | 1        | 1            |
| 1    | 39.0        | 96       | 8.5        | 71.0              | 3           | 1011.43              | 7        | 1      | 10.0            | 1        | 0            |
| 2    | 30.0        | 64       | 7.0        | 16.0              | 0           | 1018.72              | 5        | 1      | 5.5             | 2        | 3            |
| 3    | 38.0        | 83       | 1.5        | 82.0              | 0           | 1026.25              | 7        | 1      | 1.0             | 0        | 3            |
| 4    | 27.0        | 74       | 17.0       | 66.0              | 2           | 990.67               | 1        | 3      | 2.5             | 2        | 1            |

 data kategorik sudah berhasil diubah menjadi data numerik

 - Melakukan uji korelasi
   ```
    # Mengetahui skor korelasi
    plt.figure(figsize=(10, 8))
    correlation_matrix = cuaca.corr().round(2)

    # Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
    sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
    plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
    plt.tight_layout()
   ```

   output:

   ![korelasi](https://github.com/user-attachments/assets/901da1a1-4c83-409a-a41e-2478dd19e511)

  Berdasarkan nilai korelasi di atas `Temperature`, `Visibilty (km)`, dan `Location` adalah fitur yang mempunyai nilai korelasi paling kecil dengan target `Weather Type` dan akan di hapus


### Hapus Kolom dengan Korelasi Terendah

bagian ini adalah proses penghapusan fitur-fitur yang memiliki korelasi rendah terhadap variabel target dari dataset. Langkah ini diambil berdasarkan asumsi bahwa fitur dengan korelasi rendah tidak memberikan kontribusi signifikan terhadap prediksi yang dibuat oleh model.

```
cuaca.drop(['Temperature', 'Visibility (km)', 'Location'], inplace=True, axis=1)
cuaca.head()
```

output:

|      | Humidity | Wind Speed | Precipitation (%) | Cloud Cover | Atmospheric Pressure | UV Index | Season | Weather Type |
|------|----------|------------|-------------------|-------------|----------------------|----------|--------|--------------|
| 0    | 73       | 9.5        | 82.0              | 3           | 1010.82              | 2        | 3      | 1            |
| 1    | 96       | 8.5        | 71.0              | 3           | 1011.43              | 7        | 1      | 0            |
| 2    | 64       | 7.0        | 16.0              | 0           | 1018.72              | 5        | 1      | 3            |
| 3    | 83       | 1.5        | 82.0              | 0           | 1026.25              | 7        | 1      | 3            |
| 4    | 74       | 17.0       | 66.0              | 2           | 990.67               | 1        | 3      | 1            |


Penghapusan fitur `Temperature` ,`Visibilty (km)`, `Location` karena memiliki nilai korelasi yang rendah. Berdasarkan data terbaru, tersisa 8 kolom

### Train-Test-Split

Train-Test-Split adalah metode untuk membagi dataset menjadi data latih (train) dan data uji (test). Biasanya data akan dibagi dengan proporsi tertentu. Dalam kasus ini saya akan membagi data menjadi 90:10 dimana 90% untuk training dan 10% untuk testing

```
X = cuaca.drop(['Weather Type'],axis =1)
y = cuaca['Weather Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)
```

Berdasarkan output diatas kita telah sukses melakukan proses Train-Test-Split, terlihat bahwa:

- Dataset train memiliki 10520 data
- Dataset test memiliki 1169 data

### Standarisasi

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Pada kasus ini kita hanya akan melakukan standarisai pada data latih, kemudian pada tahap evaluasi kita akan melakukan standarisasi pada data uji.

```
numerical_features = ['Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()
```

output:
|      | Humidity   | Wind Speed | Precipitation (%) | Cloud Cover | Atmospheric Pressure | UV Index | Season   |
|------|------------|------------|-------------------|-------------|----------------------|----------|----------|
| 0    | -0.795473  | -1.373182  | -1.463063         | -1.919352   | 1.492121             | 0.658528 | -0.753285|
| 1    | 1.007657   | 0.315739   | 0.531440          | 0.014708    | 0.757302             | -0.728680| 0.966641 |
| 2    | -0.589401  | -0.306495  | -1.120258         | 0.014708    | 1.014102             | -0.451238| -1.613249|
| 3    | 1.059175   | -0.662057  | 1.092394          | -1.919352   | 1.706378             | 2.045736 | 0.966641 |
| 4    | -1.671279  | -1.106510  | -1.182586         | -1.919352   | 0.300168             | 0.381087 | 0.966641 |


Mengecek nilai mean dan standar deviasi setelah proses standarisasi

```
X_train[numerical_features].describe().round(4)
```

output:

|           | Humidity   | Wind Speed | Precipitation (%) | Cloud Cover | Atmospheric Pressure | UV Index | Season   |
|-----------|------------|------------|-------------------|-------------|----------------------|----------|----------|
| count     | 10520.0000 | 10520.0000 | 10520.0000        | 10520.0000  | 10520.0000           | 10520.0000| 10520.0000|
| mean      | 0.0000     | -0.0000    | -0.0000           | -0.0000     | -0.0000              | 0.0000   | 0.0000   |
| std       | 1.0000     | 1.0000     | 1.0000            | 1.0000      | 1.0000               | 1.0000   | 1.0000   |
| min       | -2.5471    | -1.6399    | -1.6189           | -1.9194     | -3.3554              | -1.0061  | -1.6132  |
| 25%       | -0.5379    | -0.7509    | -1.0579           | 0.0147      | -0.8084              | -0.7287  | -0.7533  |
| 50%       | 0.0288     | -0.1287    | 0.1263            | 0.0147      | 0.1312               | -0.4512  | 0.1067   |
| 75%       | 0.7501     | 0.7602     | 0.9054            | 0.9817      | 0.7751               | 0.6585   | 0.9666   |
| max       | 2.0380     | 2.9825     | 1.7780            | 0.9817      | 3.3214               | 2.8781   | 0.9666   |


Seperti yang disebutkan sebelumnya, proses ini akan mengubah nilai rata-rata (mean) menjadi 0 dan standar deviasi menjadi 1.

## Modeling

Pada tahap ini, saya akan mengembangkan model machine learning dengan tiga algoritma. Kemudian, saya akan mengevaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Ketiga algoritma yang akan saya gunakan, antara lain:

1. K-Nearest Neighbor
   - Kelebihan:
     - Sederhana dan mudah diimplementasikan: Tidak memerlukan asumsi distribusi data.
     - Non-parametrik: Tidak membuat asumsi tentang bentuk distribusi data.
     - Fleksibel: Dapat digunakan untuk klasifikasi dan regresi.
   - Kekurangan:
     - Lambat pada data besar: Perhitungan jarak untuk semua data memerlukan banyak waktu.
     - Sensitif terhadap skala fitur: Performa bisa terganggu jika skala fitur tidak dinormalisasi.
     - Rentan terhadap outlier: Outlier dapat mempengaruhi prediksi.
2. Random Forest
   - Kelebihan:
     - Akurasi tinggi: Menghasilkan model yang kuat melalui penggabungan banyak pohon keputusan.
     - Resisten terhadap overfitting: Karena menggunakan banyak pohon, cenderung tidak overfit.
     - Dapat menangani data yang hilang dan fitur penting: Mampu menangani data yang tidak lengkap.
   - Kekurangan:
     - Kurang interpretatif: Sulit untuk menafsirkan hasil model karena kompleksitas pohon yang dihasilkan.
     - Lambat dalam prediksi: Meskipun cepat dalam pelatihan, bisa lambat saat melakukan prediksi pada dataset besar.
     -
3. Boosting Algorithm
   - Kelebihan:
     - Akurasi sangat tinggi: Memperbaiki kesalahan dari model sebelumnya sehingga cenderung menghasilkan prediksi yang lebih akurat.
     - Bagus untuk data tidak seimbang: Dapat bekerja dengan baik pada data yang memiliki distribusi kelas yang tidak seimbang.
     - Mengurangi bias: Fokus pada kesalahan model sebelumnya mengurangi bias model.
   - Kekurangan:
     - Lebih rentan terhadap overfitting: Jika tidak diatur dengan baik, dapat menghasilkan model yang terlalu fit terhadap data pelatihan.
     - Waktu pelatihan yang lama: Karena model dilatih secara berurutan, pelatihan bisa memakan waktu lebih lama.
     - Memerlukan tuning parameter: Hyperparameter harus diatur dengan cermat untuk performa yang optimal.

Sebelum dimulainya proses modelling, mari siapkan terlebih dahulu data frame untuk analisis ketiga model tersebut.

```
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])
```

Tahap ini hanya digunakan untuk melatih data training dan menyimpan data testing dari semua model untuk tahap evaluasi yang akan dibahas di Modul Evaluasi Model

### Model K-Nearest Neighbor (KNN)

KNN adalah algoritma yang relatif sederhana dibandingkan dengan algoritma lain. Algoritma KNN menggunakan ‘kesamaan fitur’ untuk memprediksi nilai dari setiap data yang baru. Dengan kata lain, setiap data baru diberi nilai berdasarkan seberapa mirip titik tersebut dalam set pelatihan.

KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif). Nah, itulah mengapa algoritma ini dinamakan K-nearest neighbor (sejumlah k tetangga terdekat). KNN bisa digunakan untuk kasus klasifikasi dan regresi. Pada modul ini, kita akan menggunakannya untuk kasus regresi.

```
knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)
```

pada tahapan ini kita akan melatih data dengan KNN, kita menggunakan `n_neighbors`= 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik.

### Model Random Forest

Algoritma random forest adalah salah satu algoritma supervised learning. Ia dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. Random forest juga merupakan algoritma yang sering digunakan karena cukup sederhana tetapi memiliki stabilitas yang mumpuni.

Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama.

```
# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)
```

Berikut adalah parameter-parameter yang digunakan:

- `n_estimator`: jumlah trees (pohon) di forest. Di sini nilai set `n_estimator`=50.
- `max_depth`: ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Di sini nilai set `max_depth`=16.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan. Di sini nilai set `random_state`=55.
- `n_jobs`: komponen untuk mengontrol thread atau proses yang berjalan secara paralel. Di sini nilai set `n_job`s=-1 artinya semua proses berjalan secara paralel.

### Model Boosting Algorithm

Teknik boosting, model dilatih secara berurutan atau dalam proses yang iteratif. Algoritma yang menggunakan teknik boosting bekerja dengan membangun model dari data latih. Kemudian ia membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

Dilihat dari caranya memperbaiki kesalahan pada model sebelumnya, algoritma boosting terdiri dari dua metode:

1.  Adaptive boosting
2.  Gradient boosting
    Pada modul ini, kita akan menggunakan metode adaptive boosting. Salah satu metode adaptive boosting yang terkenal adalah AdaBoost, dikenalkan oleh Freund and Schapire (1995)

```
boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)
```

Berikut merupakan parameter-parameter yang digunakan pada potongan kode di atas.

- `learning_rate`: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan.

## Evaluation

Pada proses evaluasi kita akan menggunakan metrik MSE atau Mean Squared Error yang akan menghitung jumlah selisih kuadrat rata-rata nilai yang sebenarnya dengan nilai prediksi.
MSE didefinisikan dalam persamaan berikut
![MSE](https://github.com/user-attachments/assets/e16d77c4-1c0c-45b7-a7f0-ea8eb6b3d967)

Keterangan:

N = jumlah dataset

yi = nilai sebenarnya

y_pred = nilai prediksi

Namun, sebelum menghitung nilai MSE dalam model, kita perlu melakukan proses scaling fitur numerik pada data uji. Untuk proses scaling, implementasikan kode berikut:

```
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])
```

Proses scaling diatas dilakukan terhadap data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi. Selanjutnya adalah melakukan evaluasi pada ketiga model dengan metrik MSE.

```
# Buat variabel mse yang isinya adalah dataframe nilai mse data train dan test pada masing-masing algoritma
mse = pd.DataFrame(columns=['train', 'test'], index=['KNN','RF','Boosting'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung Mean Squared Error masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    mse.loc[name, 'train'] = mean_squared_error(y_true=y_train, y_pred=model.predict(X_train))/1e3
    mse.loc[name, 'test'] = mean_squared_error(y_true=y_test, y_pred=model.predict(X_test))/1e3

# Panggil mse
mse
```

output:
|           | Train      | Test       |
|-----------|------------|------------|
| KNN       | 0.000098   | 0.000131   |
| RF        | 0.000018   | 0.000080   |
| Boosting  | 0.000165   | 0.000177   |


Untuk memudahkan, mari kita plot metrik tersebut dengan bar chart. Implementasikan kode di bawah ini:

```
fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
plt.xticks(rotation=45)
```

output:
![grafik model](https://github.com/user-attachments/assets/cdfcfee8-e189-4962-9042-e01d12513fa2)

Selanjutnya kita akan melihat nilai akurasi di tiap model

```
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung akurasi masing-masing algoritma pada data test
for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    # Konversi prediksi menjadi kelas (bulatkan ke bilangan bulat terdekat)
    y_pred_class = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Akurasi {name}: {accuracy:.4f}")
```

output:

`Akurasi KNN`: 0.9145

`Akurasi RF`: 0.9461

`Akurasi Boosting`: 0.9247

Lalu Prediksi modelnya

```
# Uji data
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)
```

output:
|      | y_true | prediksi_KNN | prediksi_RF | prediksi_Boosting |
|------|--------|--------------|-------------|-------------------|
| 7259 | 2      | 1.7          | 2.0         | 1.7               |


Berdasarkan hasil akurasinya. permodelan menggunakan `K-Nearest Neighbors` mendapatkan nilai akurasi 91,45%, lalu permodelan dengan `Random Forest` mendapatkan akurasi 94,61% dan yang terakhir pada permodela `Boosting Algorithm` mendapatkan nilai akurasi 92,47%. Selain itu, hasil prediksi `K-Nearest Neighbors` dan `Random Forest` menjadi yang paling mendekati nilai sebenarnya.
Maka dari itu permodelan yang akan digunakan untuk mengklasifikasikan cuaca adalah model `Random Forest`, semoga dengan model ini bisa membantu menentukan klasifikasi cuaca yang terbaik sesuai data.

Antara UV Index dan Weather Type memiliki nilai korelasi 0,41 menunjukkan adanya korelasi positif sedang antara kedua variabel tersebut. Korelasi positif berarti bahwa ketika UV Index meningkat, kemungkinan besar Weather Type juga akan berubah ke arah yang lebih tinggi.
Dalam konteks ini, interpretasinya bisa berarti bahwa semakin tinggi nilai UV Index (yang biasanya menunjukkan sinar matahari yang lebih kuat), tipe cuaca cenderung bergerak ke arah cuaca yang lebih cerah atau lebih berpotensi terkena sinar matahari langsung. Ini masuk akal karena indeks UV biasanya lebih tinggi pada hari-hari cerah dan berkurang pada hari mendung atau hujan.
Namun, karena nilai korelasinya 0,41, ini hanya menunjukkan korelasi sedang, yang berarti UV Index adalah salah satu dari beberapa faktor yang mempengaruhi Weather Type.

Berbeda dengan UV Index, Cloud Cover memiliki nilai korelasi -0,57 antara Cloud cover (penutupan awan) dan Weather Type (tipe cuaca) menunjukkan bahwa terdapat korelasi negatif sedang antara kedua variabel tersebut. Korelasi negatif berarti bahwa ketika Cloud cover meningkat, kemungkinan besar Weather Type bergerak ke arah yang lebih rendah (atau berlawanan).
Dalam hal ini, interpretasi sederhana bisa berarti bahwa semakin tinggi penutupan awan (cuaca mendung), kemungkinan besar tipe cuaca yang berkaitan dengan cerah atau matahari akan lebih kecil, sementara tipe cuaca yang lebih mendung, hujan, atau badai lebih mungkin terjadi.

