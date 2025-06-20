import numpy as np
from collections import defaultdict

class Matrix_Factorization():
    
    def __init__(self, original_matrix, num_features, alpha, beta, iterations, train_data, test_data):
        self.original_matrix = original_matrix
        self.num_users, self.num_items = original_matrix.shape
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.train_data = train_data
        self.test_data = test_data
        
    def train(self):
        self.user_matrix = np.random.normal(scale = 1./self.num_features, size = (self.num_users, self.num_features))
        self.item_matrix = np.random.normal(scale = 1./self.num_features, size = (self.num_items, self.num_features))
        
        #Create the bias
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        
        #Find all unrated items in the original matrix
        self.b = np.mean([r for (_, _, r) in self.train_data]) if self.train_data else np.mean(self.original_matrix[self.original_matrix > 0])
        
        self.samples = self.train_data if self.train_data else [
            (i, j, self.original_matrix[i, j])
            for i in range(self.num_users)
            for j in range(self.num_items)
            if self.original_matrix[i, j] > 0
        ]        
        
        training_process = []
        
        for i in range(self.iterations):
            #Change the order of samples to ensure variation
            np.random.shuffle(self.samples)
            
            self.sgd()
            
            #Get the mean squared error
            mse = self.compute_mse()
            
            training_process.append((i, mse))
            
            if (i+1) % 20 == 0:
                print('Iteration: %d; error = %.4f' % (i+1, mse))
                
        return training_process
    
    def compute_mse(self):
        error = 0
        for i, j, r in self.samples:
            error += (r - self.get_rating(i, j)) ** 2
        
        return np.sqrt(error / len(self.samples))
    
    def compute_rmse(self):
        error = 0
        for i, j, r in self.test_data:
            pred = self.get_rating(i, j)
            error += (r - pred)**2
        
        return np.sqrt(error / len(self.test_data))
    
    def compute_recall_at_k(self, k, rating_threshold):
        #All user items that are equal to or greater than the rating threshold
        user_relevant_items = defaultdict(set)
        
        for user, item, rating in self.test_data:
            if rating >= rating_threshold:
                user_relevant_items[user].add(item)

        pred_matrix = self.full_matrix()
        total_recall = 0
        num_users_with_preds = 0

        for user, relevant_items in user_relevant_items.items():
            if len(relevant_items) == 0:
                continue

            #Identify unrated items by the user in the training dataset
            rated_items = set(np.where(self.original_matrix[user] > 0)[0])
            unrated_items = list(set(range(self.num_items)) - rated_items)

            if len(unrated_items) == 0:
                continue

            #Predict scores for unrated items
            user_predictions = pred_matrix[user, unrated_items]
            if k > len(unrated_items):
                continue

            #Get top-k recommended items
            top_k_idxs = np.argsort(user_predictions)[-k:][::-1]
            top_k_items = [unrated_items[idx] for idx in top_k_idxs]

            true_positives = len(set(top_k_items) & relevant_items)
            recall = true_positives / len(relevant_items)

            total_recall += recall
            num_users_with_preds += 1

        if num_users_with_preds > 0:
            return total_recall / num_users_with_preds
        else:
            return 0
                    
    #Gradient descent
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_rating(i, j)
            
            #Get error term for current sample
            e = (r - prediction)
            
            #Update the user bias
            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            #Update the item bias
            self.b_i[j] += self.alpha * (e - self.beta * self.b_i[j])
            
            self.user_matrix[i, :] += self.alpha * (e * self.item_matrix[j, :] - self.beta * self.user_matrix[i,:])
            self.item_matrix[j, :] += self.alpha * (e * self.user_matrix[i, :] - self.beta * self.item_matrix[j,:])
    
    #Get user i's rating for the jth item
    def get_rating(self, i, j):
        return self.b + self.b_u[i] + self.b_i[j] + self.user_matrix[i, :].dot(self.item_matrix[j, :].T)
            
    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + self.user_matrix.dot(self.item_matrix.T)
    
    
        
        
            
            
        