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
    
    
        
        
            
            
        