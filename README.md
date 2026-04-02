# 🎬 Hybrid Movie Recommendation System

A machine learning system that recommends movies using three approaches — Content-Based Filtering, Collaborative Filtering, and SVD Matrix Factorization — combined into a Hybrid model.

🚀 **Live Demo:** [Click Here](https://hybrid-recommendation-system-ix2riigq3ypvfyy3eb3jqy.streamlit.app)

---

## 📌 Problem Statement

With thousands of movies available, users struggle to find what to watch next. A recommendation system solves this by learning user preferences and suggesting relevant content — exactly how Netflix, Amazon, and Spotify work.

The challenge:
- **Cold Start Problem** — new users have no history
- **Sparsity** — most users rate very few movies
- **Scalability** — system must handle thousands of movies efficiently

---

## 🗂️ Dataset

- **Source:** [MovieLens Small Dataset](https://grouplens.org/datasets/movielens/latest/)
- **Movies:** 9,742
- **Ratings:** 100,836
- **Users:** 610
- **Rating Scale:** 0.5 to 5.0

**Files used:**

| File | Description |
|---|---|
| `movies.csv` | movieId, title, genres |
| `ratings.csv` | userId, movieId, rating, timestamp |
| `tags.csv` | userId, movieId, tag, timestamp |
| `links.csv` | movieId, imdbId, tmdbId |

---

## 🔧 Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.x |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn, SciPy |
| NLP | TF-IDF Vectorizer |
| Similarity | Cosine Similarity |
| Matrix Factorization | SVD (Singular Value Decomposition) |
| Database | MySQL |
| Deployment | Streamlit Cloud |

---

## 🧠 System Architecture

```
MovieLens Dataset
      ↓
MySQL Database (4 tables)
      ↓
Exploratory Data Analysis
      ↓
┌─────────────────────────────────────────┐
│           Three Approaches              │
│                                         │
│  1. Content-Based Filtering             │
│     TF-IDF + Cosine Similarity          │
│     (genres + user tags)                │
│                                         │
│  2. Collaborative Filtering             │
│     User-Movie Matrix                   │
│     User Similarity                     │
│                                         │
│  3. SVD Matrix Factorization            │
│     k=50 latent factors                 │
│     Predicted ratings matrix            │
└─────────────────────────────────────────┘
      ↓
Hybrid System (Content + SVD)
      ↓
Evaluation (RMSE)
      ↓
Streamlit Deployment
```

---

## 📊 How Each Approach Works

**1. Content-Based Filtering**
- Combines movie genres and user-generated tags
- Applies TF-IDF vectorization (5000 features)
- Computes cosine similarity between all movies
- Recommends movies most similar to selected movie

**2. Collaborative Filtering**
- Builds a User × Movie rating matrix
- Computes cosine similarity between users
- Finds top 5 similar users
- Recommends movies they liked that current user hasn't seen

**3. SVD (Matrix Factorization)**
- Decomposes the rating matrix into latent factors (k=50)
- Predicts ratings for all user-movie pairs
- Recommends highest predicted ratings for unseen movies

**4. Hybrid System**
- Merges Content-Based and SVD recommendations
- Returns movies that appear in both lists
- Falls back to Content-Based if no overlap found

---

## 📈 Evaluation

| Metric | Value |
|---|---|
| SVD RMSE | 2.6727 |

RMSE measures average prediction error on a 0.5–5.0 rating scale. For a baseline SVD model on MovieLens Small, this is an acceptable result.

---

## 📁 Project Structure

```
Hybrid-Recommendation-System/
│
├── recommendation.py          # Main ML pipeline
├── app_recommendation.py      # Streamlit web app
├── load_data.py               # MySQL data loader
├── requirements.txt           # Dependencies
├── README.md                  # Project documentation
│
├── movies.csv                 # Dataset
├── ratings.csv
├── tags.csv
├── links.csv
│
├── tfidf_vectorizer.pkl       # Saved TF-IDF model
├── SVD_prediction.pkl         # Saved SVD predictions
├── movies_content.pkl         # Processed movie data
└── user_movie_matrix.pkl      # User-movie rating matrix
```

---

## 🖥️ App Features

**4 tabs in the Streamlit app:**

**Tab 1 — Hybrid Recommendations**
- Select a movie + enter user ID
- Returns movies that match both content similarity and user preference
- Best overall recommendations

**Tab 2 — Content-Based**
- Select any movie
- Returns most similar movies by genre and tags
- Shows selected movie info

**Tab 3 — SVD Personalized**
- Enter user ID
- Shows rating history count
- Returns personalized predictions based on rating patterns

**Tab 4 — Dataset Explorer**
- Browse movies by genre
- Search any movie by name
- View dataset statistics

---

## ⚙️ How to Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/abenesh123/Hybrid-Recommendation-System.git
cd Hybrid-Recommendation-System
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app_recommendation.py
```

**4. Open browser at**
```
http://localhost:8501
```

---

## 🌐 Deployment

The app is deployed on **Streamlit Cloud** and publicly accessible.

🔗 **Live App:** https://hybrid-recommendation-system-ix2riigq3ypvfyy3eb3jqy.streamlit.app

---

## 📊 SQL EDA Highlights

Key queries run on MySQL database:

```sql
-- Top 10 most rated movies
SELECT m.title, COUNT(r.rating) total_ratings
FROM ratings r JOIN movies m ON r.movieId = m.movieId
GROUP BY m.title ORDER BY total_ratings DESC LIMIT 10;

-- Top rated movies (min 50 ratings)
SELECT m.title, ROUND(AVG(r.rating),2) avg_rating
FROM ratings r JOIN movies m ON r.movieId = m.movieId
GROUP BY m.title HAVING COUNT(*) >= 50
ORDER BY avg_rating DESC LIMIT 10;

-- Genre distribution
SELECT genres, COUNT(*) count FROM movies
GROUP BY genres ORDER BY count DESC LIMIT 10;
```

**Key findings:**
- Drama is the most common genre (1053 movies)
- Most users rate between 20–100 movies
- Rating distribution peaks at 4.0
- Genres are pipe-separated — split required for content analysis

---

## 💡 What I Learned

- Building three types of recommendation systems from scratch
- TF-IDF vectorization for text-based content similarity
- SVD matrix factorization for latent factor modeling
- Handling the cold start problem with hybrid approach
- Connecting MySQL database to Python
- Deploying ML apps with Streamlit Cloud
- Managing large model files with `.gitignore`

---

## 👤 Author

**Abinesh G**
- GitHub: [@abenesh123](https://github.com/abenesh123)
- LinkedIn: [abenesh-g-a94954345](https://www.linkedin.com/in/abenesh-g-a94954345)