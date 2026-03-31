import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings("ignore")

engine=create_engine("mysql+mysqlconnector://root:abineshmysql%40123@localhost/hybird_recommendation_system")

movies=pd.read_sql("SELECT * FROM movies",con=engine)
ratings=pd.read_sql("SELECT * FROM ratings",con=engine)
tags=pd.read_sql("SELECT * FROM tags",con=engine)
links=pd.read_sql("SELECT * FROM links",con=engine)

print("Data Successfully Loaded")

print(f"Movies :{movies.shape}")
print(f"Ratings :{ratings.shape}")
print(f"Tags :{tags.shape}")
print(f"Links :{links.shape}")

print(movies.head())
print(ratings.head())
print(tags.head())
print(links.head())


print(movies.info())
print(ratings.info())

print(movies.isnull().sum())
print(ratings.isnull().sum())

plt.style.use("seaborn-v0_8-darkgrid")

plt.figure(figsize=(10,5))
sns.countplot(data=ratings,x="rating",palette="Blues_r")
plt.title("Rating Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

top_rated=ratings.groupby("movieId").count()["rating"].reset_index()
top_rated.columns=["movieId","Rating_Count"]
top_rated=top_rated.merge(movies[["movieId","title"]],on="movieId")
top_rated=top_rated.sort_values("Rating_Count",ascending=False).head(10)

plt.figure(figsize=(10,5))
sns.barplot(data=top_rated,x="Rating_Count",y="title",palette="Greens_r")
plt.title("Top 10 Most Rated movie")
plt.xlabel("Number of Ratings")
plt.show()


ratings_per_userId=ratings.groupby("userId")["rating"].count()
plt.figure(figsize=(10,5))
sns.histplot(ratings_per_userId,bins=50,kde=True)
plt.title("Ratings Per UserID Distribution")
plt.xlabel("Number of Ratings")
plt.show()

movies["genres_list"]=movies["genres"].str.split("|")
from collections import Counter
all_genres=[g for sublist in movies["genres_list"] for g in sublist]
genre_count=Counter(all_genres)
genre_df=pd.DataFrame(genre_count.items(),columns=["genre","count"]).sort_values("count",ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(data=genre_df,x="count",y="genre",palette="Reds_r")
plt.title("Genre Distribution")
plt.show()

print("EDA Completed")

print("Content Based Filtering")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


movies["genres_clean"]=movies["genres"].str.replace("|"," ",regex=False)

tags_agg=tags.groupby("movieId")["tag"].apply(lambda x:" ".join(x.astype(str))).reset_index()
movies_content=movies.merge(tags_agg,on="movieId",how="left")
movies_content["tag"]=movies_content["tag"].fillna(" ")
movies_content["content"]=movies_content["genres_clean"]+ " " + movies_content["tag"]

tfidf=TfidfVectorizer(stop_words="english",max_features=5000)
tfidf_matrix=tfidf.fit_transform(movies_content["content"])
print(f"TF-IDF Shape :{tfidf_matrix}")

cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
print(f"Cosine Similarity Matrix Shape :{cosine_sim}")

indices=pd.Series(movies_content.index,index=movies_content["title"]).drop_duplicates()
print(indices)

def content_based_recommendation(title,n=10):
    if title not in indices:
        return f"Movie {title} not found"
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:n+1]
    movie_indices=[i[0] for i in sim_scores]
    result=movies_content[["title","genres"]].iloc[movie_indices].copy()
    result["silimarity_score"]=[round(i[1],3) for i in sim_scores]
    return result

print(content_based_recommendation("Toy Stor"))


print("CONTENT BASED FILTERING COMPLETED")

user_movie_matrix=ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print(user_movie_matrix)
print(f"User Movie Matrix Shape :{user_movie_matrix.shape}")

from sklearn.metrics.pairwise import cosine_similarity

user_similarity=cosine_similarity(user_movie_matrix)
print(user_similarity)

user_similarity_df=pd.DataFrame(
    user_similarity,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.index
)
print(user_similarity_df)

def collaborative_recommendation(user_id,n=10):
    if user_id not in user_similarity_df.index:
        return f"User {user_id} is not found"
    similar_users=user_similarity_df[user_id].sort_values(ascending=False)[1:6].index
    similar_users_ratings=user_movie_matrix.loc[similar_users].mean(axis=0)
    already_watched=user_movie_matrix.loc[user_id]
    already_watched=already_watched[already_watched>0].index
    recommendations=similar_users_ratings.drop(already_watched).sort_values(ascending=False).head(n)
    rec_movie=movies[movies["movieId"].isin(recommendations.index)][["movieId","title","genres",]]
    rec_movie=rec_movie.merge(recommendations.reset_index().rename(columns={0:"score"}),on="movieId").sort_values("score",ascending=False)
    return rec_movie

print(collaborative_recommendation(1))

print("COLLABORATIVE BASED FILTERING COMPLETED")

from scipy.sparse.linalg import svds
import scipy.sparse as sp

matrix=user_movie_matrix.values
u,sigma,vt=svds(matrix,k=50)
sigma=np.diag(sigma)


predicted_ratings=np.dot(np.dot(u,sigma),vt)
predicted_ratings_df=pd.DataFrame(
    predicted_ratings,
    index=user_movie_matrix.index,
    columns=user_movie_matrix.columns
)

print(f"SVD Predicted Movie Ratings :{predicted_ratings_df.shape}")

def SVD_recommendation(user_id,n=10):
    if user_id not in predicted_ratings_df.index:
        return f"User {user_id} not found"
    user_prediction=predicted_ratings_df.loc[user_id]
    already_watched=user_movie_matrix.loc[user_id]
    already_watched=already_watched[already_watched>0].index
    recommendation=user_prediction.drop(already_watched).sort_values(ascending=False).head(n)
    rec_movies=movies[movies["movieId"].isin(recommendation.index)][["movieId","title","genres"]]
    rec_movies=rec_movies.merge(recommendation.reset_index().rename(columns={user_id:"predicted_ratings"}),on="movieId").sort_values("predicted_ratings",ascending=False)
    return rec_movies

print(SVD_recommendation(1))

print("SVD RECOMMENDATION COMPLETED")


def hybird_recommendation_system(user_id,title,n=10):
    content_recs=content_based_recommendation(title,n=20)
    if isinstance(content_recs,str):
        return content_recs
    
    svd_recs=SVD_recommendation(user_id,n=20)
    if isinstance(svd_recs,str):
        return svd_recs
    

    hybird=content_recs.merge(
        svd_recs[["movieId","title","genres"]],
        left_on="title",
        right_on="title",
        how="inner"
    )


    if hybird.empty:
        return content_recs.head(n)
    
    hybird=hybird.sort_values("similarity_score",ascending=False).head(n)
    return hybird

print(hybird_recommendation_system(1,"Toy Story (1995)"))

print("HYBIRD RECOMMENDATION SYSTEM COMPLETED")

from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error

train_data,test_data=train_test_split(ratings,train_size=0.8,random_state=42)

test_preds=[]
test_actuals=[]

for _,row in test_data.iterrows():
    uid=row["userId"]
    mid=row["movieId"]
    actual=row["rating"]

    if uid in predicted_ratings_df.index and mid in predicted_ratings_df.columns:
        pred=predicted_ratings_df.loc[uid,mid]
        test_preds.append(pred)
        test_actuals.append(actual)

rmse=root_mean_squared_error(test_actuals,test_preds)        
print(f"RMSE :{rmse:.4f}")

print("EVALUATE COMPELETED")

import joblib
joblib.dump(tfidf,"tfidf_vectorizer.pkl")
joblib.dump(cosine_sim,"cosine_sim.pkl")
joblib.dump(predicted_ratings_df,"SVD_prediction.pkl")
joblib.dump(movies_content,"movies_content.pkl")
joblib.dump(user_movie_matrix,"user_movie_matrix.pkl")
print("ALL MODELS SAVED SUCCESSFULLY!")