import numpy as np

class Matrix_Factorization():
    
    def __init__(self, original_matrix, num_features, alpha, beta, iterations):
        self.original_matrix = original_matrix
        self.num_users, self.num_items = original_matrix.shape
        self.num_features = num_features
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        
    def train(self):
        self.user_matrix = np.random.normal(scale = 1./self.num_features, size = (self.num_users, self.num_features))
        self.item_matrix = np.random.normal(scale = 1./self.num_features, size = (self.num_items, self.num_features))
        
        #Create the bias
        self.b_u = np.zeros(self.num_users)
        self.b_i = np.zeros(self.num_items)
        
        #Find all unrated items in the original matrix
        self.b = np.mean(self.original_matrix[np.where(self.original_matrix != 0)])
        
        self.samples = [(i, j, self.original_matrix[i, j]) for i in range(self.num_users) for j in range(self.num_items) if self.original_matrix[i, j] > 0]
        
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
        
        #Get all non-zero elements from the original matrix
        xs, ys = self.original_matrix.nonzero()
        
        predicted = self.full_matrix()
        error = 0
        
        #Iterate over ratings and compute the mse
        for x, y in zip(xs, ys):
            
            #Get the difference between the predicted and actual value
            error += pow(self.original_matrix[x, y] - predicted[x, y], 2)
        
        #Calculate the rmse
        return np.sqrt(error)    
    
    def compute_precision_recall_at_k(self, k, rating_threshold):
        total_precision = 0
        total_recall = 0
        num_users_with_preds = 0
        
        pred_matrix = self.full_matrix()
                
        for user in range(self.num_users):
            rated_items = self.original_matrix[user] > 0
            unrated_items = ~rated_items
            
            if np.sum(unrated_items) < k:
                continue

            relevant_unrated_items = np.where((self.original_matrix[user] >= rating_threshold) & unrated_items)[0]
            num_relevant_items = len(relevant_unrated_items)
            
            if num_relevant_items == 0:
                continue
                
            user_preds = pred_matrix[user].copy()
            
            #Remove all rated items from prediction matrix
            user_preds[rated_items] = -np.diff
            
            top_k_items = np.argsort(user_preds)[-k:][::-1]
            
            num_relevant_recommended = np.sum(self.original_matrix[user, top_k_items] >= rating_threshold)
            
            precision_at_k = num_relevant_recommended / k
            recall_at_k = num_relevant_recommended / num_relevant_items

            total_precision += precision_at_k
            total_recall += recall_at_k
            num_users_with_preds += 1
        
        if num_users_with_preds == 0:
            return 0, 0

        avg_precision = total_precision / num_users_with_preds
        avg_recall = total_recall / num_users_with_preds

        return avg_precision, avg_recall
    
    def compute_recall_at_k(self, k, rating_threshold):
        total_recall = 0
        num_users_with_preds = 0
        
        predicted_matrix = self.full_matrix()
        
        for user in range(self.num_users):
            relevant_items_idxs = np.where(self.original_matrix[user, :] >= rating_threshold)[0]
            total_relevant_items = len(relevant_items_idxs)
            
            if total_relevant_items == 0:
                continue
            
            unrated_items_idxs = np.where(self.original_matrix[user, :] == 0)[0]
            
            if len(unrated_items_idxs) == 0:
                continue
            
            predicted_ratings = predicted_matrix[user, unrated_items_idxs]
            
            if k > len(unrated_items_idxs):
                continue
            
            top_k_idxs = predicted_ratings.argsort()[-k:][::-1]
            top_k_items = unrated_items_idxs[top_k_idxs]
            
            true_positives = 0
            
            for item in top_k_items:
                if item in relevant_items_idxs:
                    true_positives += 1
            
            if total_relevant_items > 0:
                recall = true_positives / total_relevant_items
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
        pred = self.b + self.b_u[i] + self.b_i[j] + self.user_matrix[i, :].dot(self.item_matrix[j, :].T)
        
        return pred
    
    def full_matrix(self):
        return self.b + self.b_u[:, np.newaxis] + self.b_i[np.newaxis, :] + self.user_matrix.dot(self.item_matrix.T)
    
    
        
        
            
            
        