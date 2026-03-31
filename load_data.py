import pandas as pd
from sqlalchemy import create_engine

movies=pd.read_csv("movies.csv")
ratings=pd.read_csv("ratings.csv")
tags=pd.read_csv("tags.csv")
links=pd.read_csv("links.csv")

print(f"Movies :{movies.shape}")
print(f"Ratings :{ratings.shape}")
print(f"Tags :{tags.shape}")
print(f"Links :{links.shape}")

engine=create_engine("mysql+mysqlconnector://root:abineshmysql%40123@localhost/hybird_recommendation_system")

movies.to_sql("movies",con=engine,if_exists="replace",index=False)
print("Movies Loaded Successfully")

ratings.to_sql("ratings",con=engine,if_exists="replace",index=False)
print("Ratings Loaded Successfully")

tags.to_sql("tags",con=engine,if_exists="replace",index=False)
print("Tags Loaded Successfully")

links.to_sql("links",con=engine,if_exists="replace",index=False)
print("Links Loaded Successfully")


import mysql.connector
conn=mysql.connector.connect(
    host="localhost",
    user="root",
    password="abineshmysql@123",
    database="hybird_recommendation_system"
)
print("connected succesfully")
conn.close()

print(f"Movies :{pd.read_sql('SELECT COUNT(*) as count FROM movies',con=engine).iloc[0,0]}")
print(f"Ratings :{pd.read_sql('SELECT COUNT(*) as count FROM ratings',con=engine).iloc[0,0]}")
print(f"Tags :{pd.read_sql('SELECT COUNT(*) FROM tags',con=engine).iloc[0,0]}")
print(f"Links :{pd.read_sql('SELECT COUNT(*) FROM links',con=engine).iloc[0,0]}")
