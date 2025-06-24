import numpy as np
from sklearn.model_selection import train_test_split
from matrix_factorization_recommendation import Matrix_Factorization
from dataframe_setup import get_dataframes

def normalize_matrix(matrix):
    user_means = np.true_divide(matrix.sum(1), (matrix != 0).sum(1))
    user_means = np.nan_to_num(user_means)  

    normalized_matrix = matrix.copy()
    for i in range(matrix.shape[0]):
        normalized_matrix[i, matrix[i] != 0] -= user_means[i]

    return normalized_matrix, user_means

def build_matrix(ratings, num_users, num_items):
    matrix = np.zeros((num_users, num_items))
    for u, i, r in ratings:
        matrix[u, i] = r
    return matrix

df_books, df_ratings, df_users = get_dataframes(sample_size=5000)  

ratings = [(row['User-ID'], row['Book-ID'], row['Book-Rating']) for _, row in df_ratings.iterrows()]
num_users = df_ratings['User-ID'].nunique()
num_items = df_ratings['Book-ID'].nunique()

train_data, test_data = train_test_split(ratings, test_size=0.2, random_state=42)
train_matrix = build_matrix(train_data, num_users, num_items)

normalized_matrix, user_means = normalize_matrix(train_matrix)

best_config = None
best_precision = 0

param_grid = {
    'alpha': [0.001, 0.005, 0.01],
    'beta': [0.01, 0.05, 0.1],
    'num_features': [20, 40, 60]
}

for alpha in param_grid['alpha']:
    for beta in param_grid['beta']:
        for num_features in param_grid['num_features']:
            print(f"Training with alpha={alpha}, beta={beta}, features={num_features}")

            mf = Matrix_Factorization(
                original_matrix=normalized_matrix,
                num_features=num_features,
                alpha=alpha,
                beta=beta,
                iterations=100,
                train_data=train_data,
                test_data=test_data
            )
            mf.train()

            #Add user means back for predictions
            full_pred_matrix = mf.full_matrix()
            for u in range(num_users):
                full_pred_matrix[u, :] += user_means[u]

            mf.full_matrix = lambda: full_pred_matrix  

            recall, precision = mf.compute_recall_precision_at_k(k=5, rating_threshold=8)

            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}\n")

            if precision > best_precision:
                best_precision = precision
                best_config = {
                    'alpha': alpha,
                    'beta': beta,
                    'num_features': num_features,
                    'precision': precision,
                    'recall': recall
                }

print("Best Config:")
print(best_config)
