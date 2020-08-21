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
import pandas as pd

#%% İMPORT TWİTTER DATA

data = pd.read_csv(r"gender_classifier.csv", encoding = "latin1")  # r = read
data = pd.concat([data.gender , data.description], axis =1 )  # concat 2 tane dataframe yi birleştirir. axis 1 sütunu alır
data.dropna(axis =0, inplace= True) # non olan satırları bul. inplace = true data nın içine yaz demek
# data = data.dropna  ile aynı şey

data.gender = [1 if i == "female" else 0 for i in data.gender]

#%% cleaning data
# regular expression 
import re

first_description = data.description[4]
description = re.sub("[^a-zA-Z]", " " ,first_description) # a-z ve A-Z arasında OLMAYANLARI (^) bul ve boşlukla değiş
description = description.lower()   # bütün harfeler küçük yapıldı

#%% stopwords (irrelavent) gereksiz kelimeler

import nltk  # natural language tool kit

nltk.download("stopwords")  # carpus diye bir klasör indiriyor
from nltk.corpus import stopwords
description = description.split()  # kelimelere ayrıldı 

description = [word for word in description if not word in set(stopwords.words("english"))]

#%%  lemmazation   kelimenin kökünü bulma

import nltk as nlp
nlp.download('wordnet')
lemma = nlp.WordNetLemmatizer()
description = [ lemma.lemmatize(i) for i in description]   # kelimenin kökü alındı

description = " ".join(description)  # splitin tam tersi kelimeleri boşluk kullanarak birleştir

#%%  ŞİMDİ BÜTÜN DATA İÇİN YAPACAĞIZ

description_list = []
for description in data.description:
    
    description = re.sub("[^a-zA-z]" , " ", description)
    description = description.lower()
    
    description = description.split()
   # description = [word for word in description if not word in set(stopwords.words("english"))]

    #lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description]
    description = " ".join(description)
    
    description_list.append(description)

#%%  Bag of words   --> En çok rastlanan kelimeleri bulduk
    
from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer(max_features= 2 ,stop_words = "english")   # 500 kelime bul , 
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray()

print("en sık kullanılan kelimeler: {}".format(count_vectorizer.get_feature_names()))  # 16224 cümle , 500 tane kelime

#%%  Naive Bayes 

y = data.iloc[: ,0].values
x = sparce_matrix

from sklearn.model_selection import train_test_split
x_train , x_test , y_train , y_test = train_test_split(x,y,test_size =0.1 , random_state =42)

from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)

y_pred = nb.predict(x_test).reshape(-1,1)
print("accuracy: ",nb.score(y_pred , y_test))

#%%


















