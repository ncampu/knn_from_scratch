# Import dependencies
import numpy as np
from collections import Counter

# Function to calculate Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1-x2)**2))

# Class for implementing the KNN algorithm
class KNN:
    def __init__(self, k=3):
        self.k = k  # Define the number of neighbors to consider

    # Store the training data
    def fit(self, X, y):
        self.X_train = X  # Features data
        self.y_train = y  # Labels

    # Predict the class for each sample in the dataset
    def predict(self, X):
        predictions = [self._predict(x) for x in X]  # Predict for each sample

        return predictions

    # Predict the class for a single sample
    def _predict(self, x):
        # Calculate the distance between the sample and all training data points
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the indices of the k closest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get the labels of the k closest neighbors
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Perform majority vote to determine the most common label among the nearest neighbors
        most_common = Counter(k_nearest_labels).most_common()

        return most_common[0][0]