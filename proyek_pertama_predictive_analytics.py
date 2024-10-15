# -*- coding: utf-8 -*-
"""Proyek Pertama: Predictive Analytics.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1kK9TZcCkz-ndg5Rg01aGiGwg2VW5dN6c

Predictive Analytics menggunakan Data Weather Type Classification

# Data Loading
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import  OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import accuracy_score

# Load data
!pip install kaggle

from google.colab import files
files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d nikhil7280/weather-type-classification
!unzip weather-type-classification.zip

# Membuat dataset bernama cuaca
cuaca = pd.read_csv('weather_classification_data.csv')
cuaca.head()

"""# Exploratory Data Analysis"""

cuaca.info()

"""Data yang digunakan berasal dari kaggle dengan judul "Weather Type Classification" yang dapat di unduh [disini](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification).

Variabel yang terdapat pada dataset adalah sebagai berikut:

- `Temperature` (numeric) : Temperatur suhu dalam celcius
- `Humidity` (numeric) : Presentase Kelembaban
- `Wind Speed` (numeric) : Kecepatan angin dalam kilometer/jam
- `Precipitation (%)` (numeric) : Presentase curah hujan
- `Cloud Cover` (categorical) : Deskripsi tutupan awan yang berisi clear, cloudy, overcast dan party cloudy
- `Atmospheric Pressure` (numeric) : Tekanan atmosfer dalam hPa
- `UV index` (numeric) : Indeks UX yang menunjukkan kekuatan radiasi UV
- `Season` (categorical) : Jenis musim mulai dari Autumn, Spring, Summer dan Winter
- `Visibility` (km) (numeric) : Jarak pandang dalam km
- `Location` (categorical) : Lokasi dimana data di ambil seperti coastal, inland dan muntain
- `Weather Type` (categorical) : Jenis cuaca yang berisi Cloudy, Rainy, Snowy dan Sunny (Target Klasifikasi)

Totalnya ada 11 variabel dengan jumlah 13200 data
"""

# Cek nilai duplikat pada data
duplicate_rows = cuaca[cuaca.duplicated()]
print("Jumlah baris duplikat:", duplicate_rows.shape[0])

"""Berdasarkan hasil pengecekan tidak ditemukan nilai duplikat"""

# Cek nilai kosong pada data
print(cuaca.isnull().sum())

"""Berdasarkan pengecekan juga, tidak ditemukan data yang kosong

## Univariate Analysis

Univariate Analysis adalah jenis analisis data yang memeriksa satu variabel saja. Tujuannya uuntuk menggambarkan data dan menemukan pola distribusi data

Sebelum mulai analysis kita bagi datanya menjadi 2 bagian, yakni `numerical_fitur` untuk data numerik dan `categorical_features` untuk data kategorik
"""

# bagi menjadi 2 fitur
numerical_features = ['Temperature', 'Humidity', 'Wind Speed', 'Precipitation (%)', 'Atmospheric Pressure', 'UV Index', 'Visibility (km)']
categorical_features = ['Cloud Cover', 'Season', 'Location', 'Weather Type']

"""### Categorical Features"""

# Fitur CLoud Cover
feature = categorical_features[0]
count = cuaca[feature].value_counts()
percent = 100*cuaca[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Berdasarkan grafik pada fitur `Cloud Cover` di atas:
- `overcast` memiliki 6090 data
- `party cloud` memiliki 4560 data
- `clear` memiliki 2139 data
- `cloudy` memiliki 411 data
"""

# Fitur Season
feature = categorical_features[1]
count = cuaca[feature].value_counts()
percent = 100*cuaca[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Berdasarkan grafik pada fitur `Season` di atas:
- `winter` memiliki 5610 data
- `Spring` memiliki 2598 data
- `Autumn` memiliki 2500 data
- `Summer` memiliki 2492 data
"""

# Fitur Location
feature = categorical_features[2]
count = cuaca[feature].value_counts()
percent = 100*cuaca[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Berdasarkan grafik pada fitur `Location` di atas:
- `inland` memiliki 4816 data
- `mountain` memiliki 4813 data
- `coastal` memiliki 3571 data
"""

# Fitur Weather Type
feature = categorical_features[3]
count = cuaca[feature].value_counts()
percent = 100*cuaca[feature].value_counts(normalize=True)
df = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Berdasarkan grafik pada fitur `Weather Type` di atas, nilai pada kolom ini terlihat seimbang dengan rincian sebagai berikut:
- `Rainy` memiliki 3300 data
- `Cluody` memiliki 3300 data
- `Sunny` memiliki 3300 data
- `Snowy` memiliki 3300 data

### Numerical Features
"""

cuaca.hist(bins=50, figsize=(20,15))
plt.show()

"""Berdasarkan grafik diatas, hampir semmua kolom skewnessnya mengarah ke kiri kecuali `Humidity`, `Precipitaion (%)` dan `Atmospheric Pressure`.

## Multivariate Analysis

Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel pada data. Multivariate Analysis yang menunjukkan hubungan antara dua variabel biasa disebut sebagai bivariate Analysis. Selanjutnya, kita akan melakukan analisis data pada fitur kategori dan numerik.

### Categorical Features
"""

for feature in categorical_features[:-1]:
  plt.figure(figsize=(10, 6))
  sns.countplot(x=feature, hue='Weather Type', data=cuaca)
  plt.title(f'Hubungan {feature} dengan Weather Type')
  plt.xlabel(feature)
  plt.ylabel('Jumlah Data')
  plt.legend(title='Weather Type')
  plt.show()

"""Berdasarkan grafik di atas didapatn:
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

### Numerical Features
"""

sns.pairplot(cuaca, hue='Weather Type')
plt.show()

"""Berdasarkan visualisasi data diatas, tidak terlihat adanya hubungan yang signifikan antara fitur dengan target `Weather Type`

# Data Preparation

Data preparation merupakan tahapan penting dalam proses pengembangan model machine learning. Ini adalah tahap di mana kita melakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan.

## Menangani Outliers

Outlier adalah titik data yang secara signifikan berada di sebgaian data dalam kumpulan data. Outlier ini bisa muncul karena banyak faktor salah satunya adalah kesalahan pengamatan.
"""

#menampilkan data outlier
for column in cuaca.select_dtypes(include=np.number).columns:
  plt.figure(figsize=(8, 6))
  sns.boxplot(x=cuaca[column])
  plt.title(f'Boxplot of {column}')
  plt.show()

"""Berdasarkan boxplot diatas, ada 4 fitur yang memiliki outlier yakni fitur `Temperature`, `Wind Speed`, `Athmospheric Pressure` dan `Visibility (km)`

Outlier perlu dihapus untuk mendapatkan model yang bagus

"""

# Pilih yang numerik saja
numeric_cuaca = cuaca.select_dtypes(include=np.number)

Q1 = numeric_cuaca.quantile(0.25)
Q3 = numeric_cuaca.quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

cuaca = cuaca[~((numeric_cuaca < lower_bound) | (numeric_cuaca > upper_bound)).any(axis=1)]

"""Outlier telah dihapus"""

cuaca.shape

"""Jumlah data sekarang menjadi 11689 dari 13200 data

## Mengubah Type data
"""

# ubah semua data kategori menjadi numerik
le = LabelEncoder()
cuaca['Cloud Cover'] = le.fit_transform(cuaca['Cloud Cover'])
cuaca['Location'] = le.fit_transform(cuaca['Location'])
cuaca['Season'] = le.fit_transform(cuaca['Season'])
cuaca['Weather Type'] = le.fit_transform(cuaca['Weather Type'])
cuaca.head()

"""- Mengubah semua data kategorik menjadi numerik untuk mempermudah permodelan"""

# update data
cuaca.describe()

# cek data lagi
cuaca.info()

# Mengetahui skor korelasi
plt.figure(figsize=(10, 8))
correlation_matrix = cuaca.corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)
plt.tight_layout()

"""Berdasarkan nilai korelasi di atas
- `Temperature`, `Visibilty (km)`, dan `Location` adalah fitur yang mempunyai nilai korelasi paling kecil dengan target `Weather Type` dan akan di hapus

## Hapus Kolom dengan Korelasi Terendah

bagian ini adalah proses penghapusan fitur-fitur yang memiliki korelasi rendah terhadap variabel target dari dataset. Langkah ini diambil berdasarkan asumsi bahwa fitur dengan korelasi rendah tidak memberikan kontribusi signifikan terhadap prediksi yang dibuat oleh model.
"""

# Ada beberapa yang tidak memilik korelasi dengan Weather Type, maka dihilangkan
cuaca.drop(['Temperature', 'Visibility (km)', 'Location'], inplace=True, axis=1)
cuaca.head()

cuaca.info()

"""Penghapusan fitur `Temperature` ,`Visibilty (km)`, `Location` karena memiliki nilai korelasi yang rendah. Berdasarkan data terbaru, tersisa 8 kolom

## Train-Test-Split

Train-Test-Split adalah metode untuk membagi dataset menjadi data latih (train) dan data uji (test). Biasanya data akan dibagi dengan proporsi tertentu. Dalam kasus ini saya akan membagi data menjadi 90:10 dimana 90% untuk training dan 10% untuk testing
"""

# Membagi 90:10 (10% untuk data uji/test)
X = cuaca.drop(['Weather Type'],axis =1)
y = cuaca['Weather Type']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

# cek jumlah sampel
print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""Berdasarkan output diatas kita telah sukses melakukan proses Train-Test-Split, terlihat bahwa:
- Dataset train memiliki 10520 data
- Dataset test memiliki 1169 data

## Standarisasi

Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk fitur numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi.  StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

Pada kasus ini kita hanya akan melakukan standarisai pada data latih, kemudian pada tahap evaluasi kita akan melakukan standarisasi pada data uji.
"""

# Standarisasi data latih (train) dengan StandardCaler
numerical_features = ['Humidity', 'Wind Speed', 'Precipitation (%)', 'Cloud Cover', 'Atmospheric Pressure', 'UV Index', 'Season']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

# mengecek nilai mean dan standar deviasi pada setelah proses standarisasi
X_train[numerical_features].describe().round(4)

"""Seperti yang disebutkan sebelumnya, proses ini akan mengubah nilai rata-rata (mean) menjadi 0 dan standar deviasi menjadi 1.

# Model Deployment

Pada tahap permodelan ini saya akan menggunakan 3 model yang berbeda, berikut ini adalah ketiga algoritma tersebut:
1. K-Nearest Neighbor (KNN)
  - Kelebihan
    - Sederhana dan mudah diimplementasikan
    - Non-parametrik
    - Fleksibel
  - Kekurangan
    - Lambat pada data besar
    - Sensitif terhadap fitur skala
    - Rentan terhadap outlier

2. Random Forest (RF)
  - Kelebihan
    - Akurasi tinggi
    - Resisten terhadap overfitting
    - Dapat menangani data yang hilang dan fitur penting
  - Kekurangan
    - Kurang interpretatif
    - Lambat dalam prediksi

3. Boosting Algorithm
  - Kelebihan
    - Akurasi sangat tinggi
    - Bagus untuk data tidak seimbang
    - Mengurangi bias
  - Kekurangan
    - Lebih rentan terhadap overfitting
    - Waktu pelatihan yang lama
    - Memerlukan tuning parameter

Sebelum kita mulai proses modellingnya, mari siapkan data frame untuk analisis ketiga model tersebut lebih dahulu
"""

# Siapkan dataframe untuk analisis model
models = pd.DataFrame(index=['train_mse', 'test_mse'],
                      columns=['KNN', 'RandomForest', 'Boosting'])

"""Pada tahap ini kita hanya melatih data training dan menyimpan data testing dari semua model untuk tahap evaluasi yang akan dibahas di Modul Evaluasi Model

## Model K-Nearest Neighbor (K-NN)
"""

knn = KNeighborsRegressor(n_neighbors=10)
knn.fit(X_train, y_train)

models.loc['train_mse','knn'] = mean_squared_error(y_pred = knn.predict(X_train), y_true=y_train)

"""pada tahapan ini kita akan melatih data dengan KNN, kita menggunakan `n_neighbors`= 10 tetangga dan metric Euclidean untuk mengukur jarak antara titik.

## Model Random Forest
"""

# buat model prediksi
RF = RandomForestRegressor(n_estimators=50, max_depth=16, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)

models.loc['train_mse','RandomForest'] = mean_squared_error(y_pred=RF.predict(X_train), y_true=y_train)

"""Berikut adalah parameter-parameter yang digunakan:

- `n_estimator`: jumlah trees (pohon) di forest. Di sini kita set `n_estimator`=50.
- `max_depth`: ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan. Di sini kita set `max_depth`=16.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan. Di sini kita set `random_state`=55.
- `n_jobs`:  komponen untuk mengontrol thread atau proses yang berjalan secara paralel.  Di sini kita set `n_job`s=-1 artinya semua proses berjalan secara paralel.

## Model Boosting Algorithm
"""

boosting = AdaBoostRegressor(learning_rate=0.05, random_state=55)
boosting.fit(X_train, y_train)
models.loc['train_mse','Boosting'] = mean_squared_error(y_pred=boosting.predict(X_train), y_true=y_train)

"""kita akan menggunakan metode adaptive boosting. Salah satu metode adaptive boosting yang terkenal adalah AdaBoost.

Berikut merupakan parameter-parameter yang digunakan pada potongan kode di atas.

- `learning_rate`: bobot yang diterapkan pada setiap regressor di masing-masing proses iterasi boosting.
- `random_state`: digunakan untuk mengontrol random number generator yang digunakan.

# Evaluasi Model

Pada proses evaluasi kita akan menggunakan metrik MSE atau Mean Squared Error yang akan menghitung jumlah selisih kuadrat rata-rata nilai yang sebenarnya dengan nilai prediksi.

Namun, sebelum menghitung nilai MSE dalam model, kita perlu melakukan proses scaling fitur numerik pada data uji
"""

# Proses Scalling
# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

"""Proses scaling diatas dilakukan terhadap data uji. Hal ini harus dilakukan agar skala antara data latih dan data uji sama dan kita bisa melakukan evaluasi.

Selanjutnya adalah melakukan evaluasi pada ketiga model dengan metrik MSE.
"""

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

fig, ax = plt.subplots()
mse.sort_values(by='test', ascending=False).plot(kind='barh', ax=ax, zorder=3)
ax.grid(zorder=0)
plt.xticks(rotation=45)

"""Selanjutnya kita akan melihat nilai akurasi di tiap model"""

# melihat nilai akurasi dari tiap model
# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'KNN': knn, 'RF': RF, 'Boosting': boosting}

# Hitung akurasi masing-masing algoritma pada data test
for name, model in model_dict.items():
    y_pred = model.predict(X_test)
    # Konversi prediksi menjadi kelas (bulatkan ke bilangan bulat terdekat)
    y_pred_class = np.round(y_pred).astype(int)
    accuracy = accuracy_score(y_test, y_pred_class)
    print(f"Akurasi {name}: {accuracy:.4f}")

"""Berdasarkan visualisasi dan nilai akurasi pada ketiga model. Kita mendapatkan nilai tertinggi pada `Random Forest` dengan akurasi 94.61%.

Selanjutnya kita uji prediksinya menggunakan beberapa nilai dalam data
"""

# Uji data
prediksi = X_test.iloc[:1].copy()
pred_dict = {'y_true':y_test[:1]}
for name, model in model_dict.items():
    pred_dict['prediksi_'+name] = model.predict(prediksi).round(1)

pd.DataFrame(pred_dict)

"""Berdasarkan prediksinya juga, `Random forest` memiliki hasil prediksi terbaik"""