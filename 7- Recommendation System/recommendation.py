
import pandas as pd
import os

#%% import movies

movie = pd.read_csv("../input/movie.csv")
movie.columns 
# (["movieID", "title" , "genres"])
movie = movie.loc[: , ["movieId", "title"]]
movie.head(10)     # ilk 10 satırı al

#%% import rating

rating = pd.read_csv("../input/rating.csv")
rating = rating.loc[:, ["userId","movieId", "rating" ]]
rating.head(10)

#%%  movies + rating

data = pd.merge(movie,rating)  # iki dataframe yi birleştirdik
data.head(10)
data = data.iloc[:1000000 , :]  # ilk 1000000 film üzerinden yapacağız.

#%%   Bad Boys filmine en yakın filmi bul

movie_watched = pivot_table["Bad Boys (1995)"]   # bad boys sütununu aldık
similarity_with_other_movies = pivot_table.corrwith(movie_watched) # diğer sütunlarla tek tek karşılaştır
# benzerse 1 e yakın benzemezse 0 a yakın verir. Ancak sırayla karşılaştırıyor
similarity_with_other_movies = similarity_with_other_movies.sort_values(ascending = False) # büyükten küçüğe sıralar
