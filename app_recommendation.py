import streamlit as st
import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(
    page_title="Movie Recommendation System",
    page_icon="🎬",
    layout="wide"
)

@st.cache_resource
def load_model():
    tfidf=joblib.load("tfidf_vectorizer.pkl")
    svd_preds=joblib.load("SVD_prediction.pkl")
    movies_content=joblib.load("movies_content.pkl")
    user_movie_matrix=joblib.load("user_movie_matrix.pkl")
    tfidf_matrix=tfidf.transform(movies_content["title"])
    cosine_sim=cosine_similarity(tfidf_matrix,tfidf_matrix)
    return tfidf,cosine_sim,svd_preds,movies_content,user_movie_matrix

tfidf,cosine_sim,svd_preds,movies_content,user_movie_matrix=load_model()

indices=pd.Series(
    movies_content.index,
    index=movies_content["title"]
).drop_duplicates()

def content_based_recommendation(title,n=10):
    if title not in indices:
        return None
    idx=indices[title]
    sim_scores=list(enumerate(cosine_sim[idx]))
    sim_scores=sorted(sim_scores,key=lambda x:x[1],reverse=True)
    sim_scores=sim_scores[1:n+1]
    movies_indices=[i[0] for i in sim_scores]
    result=movies_content[["title","genres"]].iloc[movies_indices].copy()
    result["similarity_score"]=[round(i[1],3) for i in sim_scores]
    result=result.reset_index(drop=True)
    result.index += 1
    return result


def svd_recommendation(user_id,n=10):
    if user_id not in svd_preds.index:
        return None
    user_prediction=svd_preds.loc[user_id]
    already_watched=user_movie_matrix.loc[user_id]
    already_watched=already_watched[already_watched>0].index
    recommendation=user_prediction.drop(already_watched).sort_values(ascending=False).head(n)
    movies_df=movies_content[["movieId","title","genres"]].drop_duplicates("movieId")
    rec_movies=movies_df[movies_df["movieId"].isin(recommendation.index)].copy()
    rec_movies=rec_movies.merge(
        recommendation.reset_index().rename(columns={user_id:"predicted_ratings"}),on="movieId").sort_values("predicted_ratings",ascending=False)
    rec_movies=rec_movies.reset_index(drop=True)
    rec_movies.index += 1
    return rec_movies[["title","genres","predicted_ratings"]]


def hybird_recommendation_system(user_id,title,n=10):
    content_rec=content_based_recommendation(title,n=30)
    svd_rec=svd_recommendation(user_id,n=30)

    if content_rec is None:
        return None, "content"
    if svd_rec is None:
        return content_rec.head(n),None
    
    hybird=content_rec.merge(
        svd_rec[["title","predicted_ratings"]],
        on="title",
        how="inner"
    )


    if hybird.empty:
        return content_rec.head(n),None
    hybird=hybird.sort_values("similarity_score",ascending=False).head(n)
    hybird=hybird.reset_index(drop=True)
    hybird.index += 1
    return hybird,None




st.title("🎬 Hybird Movie Recommendation System")
st.markdown("Get personalized movie recommendations using Content-Based, Collaborative, and SVD filtering.")
st.divider()

with st.sidebar:
    st.header("ℹ️ About")
    st.write("This system recommends movies using 3 approaches combined into a Hybrid model.")
    st.divider()
    st.header("📊 System Info")
    st.write(f"**Total Movies:** {len(movies_content['title'].unique()):,}")
    st.write(f"**Total Users:** {len(user_movie_matrix):,}")
    st.write(f"**SVD RMSE:** 2.6727")
    st.divider()
    st.header("🔍 How It Works")
    st.write("**Content-Based:** Recommends similar movies based on genres and tags")
    st.write("**Collaborative:** Finds users with similar taste and recommends their movies")
    st.write("**SVD:** Matrix factorization to find hidden patterns in ratings")
    st.write("**Hybrid:** Combines Content + SVD for best results")
    st.divider()
    st.header("📁 Dataset")
    st.write("**Source:** MovieLens Small")
    st.write("**Ratings:** 100,836")
    st.write("**Movies:** 9,742")
    st.write("**Users:** 610")

tab1,tab2,tab3,tab4=st.tabs([
    "🎯 Hybrid Recommendations",
    "🎥 Content-Based",
    "👥 SVD (User-Based)",
    "📊 Dataset Explorer"
])

with tab1:
    st.subheader("Get Hybrid Recommendation")
    st.markdown("Enter a movie you like and your user ID to get personalized recommendation.")

    col1,col2=st.columns(2)

    with col1:
        all_titles=sorted(movies_content["title"].unique().tolist())
        selected_movie=st.selectbox("Select a movie you like",options=all_titles)

    with col2:
        user_id_hybird=st.number_input("Enter your user ID (1-610)",min_value=1,max_value=610,value=1)    

    n_hybird=st.slider("Number of Recommendation",min_value=5,max_value=20,value=10)    



    if st.button("🎯 Get Hybird Recommendation",use_container_width=True):
        with st.spinner("Finding Best Movies For You..."):
            results,error=hybird_recommendation_system(user_id_hybird,selected_movie,n=n_hybird)
        if error=="content":
            st.error(f"Movie {selected_movie} is not found in database")
        elif results is not None and not results.empty:
            st.success(f"Top {n_hybird} hybrid recommendations based on **{selected_movie}** for User **{user_id_hybird}**")     
            st.dataframe(results,use_container_width=True)   
        else:
            st.warning("No hybrid results found. Try a different movie or user ID.")        


with tab2:
    st.subheader("Get Content-Based Recommendations")
    st.markdown("Recommend moives similar to one you selected based on **genres and tags**,")

    all_titles_cb=sorted(movies_content["title"].unique().tolist())
    selected_movie_cb=st.selectbox("Select a Movie",options=all_titles_cb,key="cb")
    n_cb=st.slider("Number of Recommendations",min_value=5,max_value=20,value=10)


    if st.button("🎥 Get Content-Based Recommendations",use_container_width=True):
        with st.spinner("Finding similar movies..."):
           results_cb=content_based_recommendation(selected_movie_cb,n=n_cb)

        if results_cb is None:
            st.error(f"Movie '{selected_movie_cb}' not found.")
        else:
            st.success(f"Movie sinilar to '{selected_movie_cb}'")     

            col_a,col_b=st.columns([2,1])

            with col_a:
                st.dataframe(results_cb,use_container_width=True)

            with col_b:
                st.subheader("Selected Movie INFO")
                movie_info=movies_content[movies_content["title"]==selected_movie_cb].iloc[0]
                st.write(f"**Title:** {movie_info["title"]}")
                st.write(f"**Genres:** {movie_info["genres"]}") 

with tab3:
    st.subheader("SVD - Personalized Recommendation")
    st.markdown("Recommend movies based on your **ratings history** using matrix factorization")

    user_id_svd=st.number_input("Enter user ID (1-610)",min_value=1,max_value=610,value=1,key="svd_user")
    n_svd=st.slider("Number of Recommendations",min_value=5,max_value=20,value=10,key="svd_n")


    if st.button("👥 Get SVD Recommendations",use_container_width=True):
        with st.spinner("Calculating Personalized Recommendation"):
            results_svd=svd_recommendation(user_id_svd,n=n_svd)

            if results_svd is None:
                st.error(f"user {user_id_svd} is not found")

            else:
                user_rated=user_movie_matrix.loc[user_id_svd]
                user_rated=user_rated[user_rated>0]    
                num_ratings=len(user_rated)

                col1,col2,col3=st.columns(3)
                col1.metric("User ID", user_id_svd)
                col2.metric("Movies Rated", num_ratings)
                col3.metric("recommendations", n_svd)

                st.divider()
                
                st.success(f"Top {n_svd} personalized recommendations for User **{user_id_svd}**")
                st.dataframe(results_svd,use_container_width=True)

with tab4:
    st.subheader("Dataset Explorer")
    col1,col2,col3=st.columns(3)
    col1.metric("Total Movies", f"{len(movies_content["title"].unique()):,}")
    col2.metric("Total Users", f"{len(user_movie_matrix):,}")
    col3.metric("SVD RMSE", "2.6727")

    st.divider()

    st.subheader("Browse Movies by Genres")
    all_genres=set()
    for g in movies_content["genres"].str.split("|"):
        all_genres.update(g)
    all_genres=sorted(list(all_genres))    

    selected_genres=st.selectbox("Select Genres",options=["ALL"] + all_genres)

    if selected_genres=="ALL":
        display_movies=movies_content[["title","genres"]].drop_duplicates("title")
        st.write(f"Showing **{len(display_movies)}** movies")
        st.dataframe(display_movies.reset_index(drop=True), use_container_width=True)

    else:
        display_movies=movies_content[movies_content["genres"].str.contains(selected_genres,na=False)][["title","genres"]].drop_duplicates("title")
        st.write(f"Showing **{len(display_movies)}** movies")    
        st.dataframe(display_movies.reset_index(drop=True),use_container_width=True)

        st.divider()

    st.subheader("Search Movies")
    search_query=st.text_input("Type Movie Name to Search")

    if search_query:
        search_results=movies_content[movies_content["title"].str.contains(search_query,case=False,na=False)][["title","genres"]].drop_duplicates("title")

        st.write(f"Found **{len(search_results)}** Movies")
        st.dataframe(search_results.reset_index(drop=True),use_container_width=True)    