# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import streamlit as st
import re

books_filename   = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'

# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author', 'year','publisher','image_s','image_m','image_l'],
    usecols=['isbn', 'title', 'author','year','publisher','image_s','image_m','image_l'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str','year': 'str',
           'publisher': 'str','image_s':'str','image_m':'str',
           'image_l':'str'})


df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

df_books.dropna(inplace=True)

sp5 = df_ratings.shape

ratings = df_ratings['user'].value_counts()
# print(len(ratings[ratings < 200]))
df_ratings['user'].isin(ratings[ratings < 200].index).sum()

df_ratings_rm = df_ratings[
  ~df_ratings['user'].isin(ratings[ratings < 200].index)
]
sp1 = df_ratings_rm.shape

ratings = df_ratings['isbn'].value_counts()

# print(len(ratings[ratings < 100]))

df_ratings_rm = df_ratings_rm[
  ~df_ratings_rm['isbn'].isin(ratings[ratings < 100].index)
]
sp2 = df_ratings_rm.shape

# books = ["Where the Heart Is (Oprah's Book Club (Paperback))",
#         "I'll Be Seeing You",
#         "The Weight of Water",
#         "The Surgeon",
#         "I Know This Much Is True"]

# for book in books:
#   print(df_ratings_rm.isbn.isin(df_books[df_books.title == book].isbn).sum())
  
df = df_ratings_rm.pivot_table(index=['user'],columns=['isbn'],values='rating').fillna(0).T

df.index = df.join(df_books.set_index('isbn'))['title']
df = df.sort_index()

model = NearestNeighbors(metric='cosine')
mp1= model.fit(df.values)

sp4 = df.iloc[0].shape

title = 'The Queen of the Damned (Vampire Chronicles (Paperback))'
sp3= df.loc[title].shape

distance, indice = model.kneighbors([df.loc[title].values], n_neighbors=6)

# print(distance)
# print(indice)


def get_recommends(title = ""):
  try:
    book = df.loc[title]
  except KeyError as e:
    print('The given book', e, 'does not exist')
    return
  
  books_val = book.values
  if book.values.ndim>1:
    books_val= book.values.transpose()[:,1]
  distance, indice = model.kneighbors([books_val], n_neighbors=6)

  recommended_books = pd.DataFrame({
      'title'   : df.iloc[indice[0]].index.values,
      'distance': distance[0]
    }) \
    .sort_values(by='distance', ascending=False) \
    .head(5).values

  return [title, recommended_books]

st.title("Book Recommendation system using KNN")
title = st.selectbox("Enter book", list(df.index.values))
# title = st.text_input('Book title', "The Queen of the Damned (Vampire Chronicles (Paperback))")
recom = get_recommends(title)
if recom:
    list_col = st.columns(len(recom[1]))

    i=0
    for r in recom[1]:
        image = list(df_books[df_books.title == r[0]].image_l)[0]
        with list_col[i]:
            st.write(re.sub(r"\(.*\)",'',r[0]))
            st.image(image)
        i += 1
else:
    st.write("No books found for recommendation")