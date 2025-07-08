📚 Hybrid Recommender Systems – Books & Movies
Built both collaborative and content-based recommender systems using Python 3.11 to explore and compare different recommendation techniques on large-scale datasets.

📘 Collaborative Filtering (Books)
Utilized a massive book rating dataset; applied custom resampling strategies to filter for active users and frequently reviewed books due to hardware constraints.

Developed multiple models:

Baseline: User-item interaction matrix with cosine similarity.

Advanced: K-Nearest Neighbors and SVD-based collaborative filtering.

Matrix Factorization with SGD: Tuned hyperparameters to optimize Recall@K, Precision@K, and RMSE. Despite attempts to reduce latent factors, model quality suffered due to hardware limitations.

Techniques: Matrix factorization, cosine similarity, KNN, SVD, SGD, normalization, resampling, evaluation metrics (Recall@K, Precision@K, RMSE), hyperparameter tuning.

🎬 Content-Based Filtering (Movies)
Built a lightweight movie recommender for concept exploration.

Used textual metadata (overview, genre, director, etc.) to compute similarity.

Preprocessing: Text cleaning, lemmatization, min-max normalization.

Used SentenceTransformers (all-MiniLM-L6-v2) for semantic embeddings; computed cosine similarity to return top 10 similar movies.

This project served as both a learning platform and a technical showcase, balancing practical system limitations with theoretical application.
