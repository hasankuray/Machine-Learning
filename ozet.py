# GORUNTULEME

import matplotlib.pyplot as plt

plt.scatter(x,y,color = "red") # x ve y beraber olacak şekilde noktalar belirler

plt.plot(x,y, color = "red" , label = "etiket")   # çizgi çeker

plt.show()   # herşeyi göstermek için

plt.xlabel("x in değerleri")
plt.ylabel("y nin değerleri")

plt.legend()  #  etiketleri göstermek için kullanılır

#%%           REGRESSİON 

# Adımlar
"""
1- Bir tane dataframe tanımla
2- x ve y eksenlerimizi tanımla
3- modeli fit et
4- x değerlerini predict ederek y_head bul
"""

# Komutlar

df = pd.read_csv("csv_dosyası.csv", sep = ";")

x = df.deneyim.values.reshape(-1,1)

x = df.iloc[: , [0,2]].values   # her satır alonsın , 0 ve 2 sütunlar alınsın

x = np.arange(min(x) , max(x) , 0.01)  # min ile max değer arasında 0.01 ilerler ve bir dizi oluşturur.

data.drop(["istenmeyen1"], axis =1 , inplace=True) # axis 0 ise satır 1 ise sütun // inplace true ise dataya kaydet

x = (y - np.min(y))  /  (np.max(y) - np.min(y))  # normalize 

x = x.T   # Matristeki sütunlar ve satırları yer değiştirir.

#%%     CLASSİFİCATION

M = data[data.diagnosis == "M"]  # data.diagnosis i M olan bütün dataları M ye at

#%%     CLUSTERİNG 

x1 = np.random.normal(25,5,1000) # ortalaması 25 olacak , 1000 tane sayı , sayıların %66 sı 25-5 ve 25+5 arasında.

x = np.concatenate((x1,x2,x2) , axis =0)  # x1 , x2 , x3 ü axis 0 da yukardan aşağı birleştirdik.

dictionary = {"x" : x , "y":y}

data.describe()   # datanın özelliklerini gösterir

#%%     NLP (Natural Language Process)

# Adımlar
"""
1- Datayı tanımla
2- Boş sütun veya boş cümleler varsa onları çıkar
3- Gereksiz işaretleri çıkar
4- Bütün harfleri küçük yap
5- Cümleyi kelimelere ayır
6- Kelimenin kökünü al
7 -Cümleyi birleştir
8- Bag of words ile en çok kullanılan kelimeleri bul
9- Machine Learning algoritması kullan
"""

data = pd.concat([data.gender , data.description], axis =1 )  # 2 tane dataframe yi birleştirir. Axis 1 sütunu alır

data.dropna(axis =0, inplace= True) # non olan satırları bul. inplace = true data nın içine yaz demek

#%%     PCA

df = pd.DataFrame(data , columns = sütun_isimleri)

df["sinif"] = y    # sınıf diye sütun ekler ve y değerlerini içine atar

#%%    RECOMMENDATİON SYSTEM

movie = movie.loc[: , ["movieId", "title"]]

data = pd.merge(movie,rating)  # iki dataframe yi birleştirdik

movie.head(10)     # ilk 10 satırı al
















