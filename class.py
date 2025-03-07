import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class ADALine():
    
    def __init__(self, train_data, test_data, lr=0.0025, weights=None):
        self.lr = lr
        self.min_error = 1e-6
        self.norm_params = None
        self.inputs = np.array(train_data.drop('d', axis=1), dtype=float)
        self.outputs = np.array(train_data['d'], dtype=float)
        self.test_data = np.array(test_data, dtype=float)
        self.input_size = self.inputs.shape[0]
        self.num_samples = self.inputs.shape[1]
        self.weights = weights


    def _get_normalization_params(self, data):
        return {
            'min': data.min(),
            'max': data.max()
        }
    

    def _normalize(self, data):
        if self.norm_params is None:
            self.norm_params = self._get_normalization_params(data)
        return (data - self.norm_params['min']) / (self.norm_params['max'] - self.norm_params['min'])


    def _insert_threshold(self, data):
        data = np.insert(data, 0, -1, axis=1).transpose() # transpose to get the right shape
        self.input_size, self.num_samples = data.shape[0], data.shape[1]
        return data


    def mean_squared_error(self):
        return np.sum((self.outputs - np.dot(self.weights, self.inputs)) ** 2) / self.num_samples
    

    def initialize_weights(self):
        self.weights = np.random.rand(self.input_size).reshape(1, self.input_size)


    def pre_processing(self, data):
        data = self._normalize(data)
        data = self._insert_threshold(data)
        return data

    
    def delta(self):
        for i in range(self.num_samples):
            print('rodando')
            activation_potential = np.dot(self.weights, self.inputs[:, i])
            self.weights = self.weights + self.lr * (self.outputs[i] - activation_potential) * self.inputs[:, i]


    def train(self):
        self.inputs = self.pre_processing(self.inputs)
        self.initialize_weights()
        num_epochs = 0
        print('Initial weights:', self.weights)
        while True:
            previous_eqm = self.mean_squared_error()
            self.delta()
            current_eqm = self.mean_squared_error()
            num_epochs +=1
            if abs(current_eqm - previous_eqm) < self.min_error:
                print('Final weights:', self.weights)
                print('Number of epochs:', num_epochs)
                break
    

    def predict(self, data):
        data = self.pre_processing(data)
        for i in range(data.shape[1]):
           print('Class:', 1 if np.dot(self.weights, data[:, i]) > 0 else -1)
    
    
if __name__ == '__main__':
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')
    adaline = ADALine(train_data, test_data)
    adaline.train()
    adaline.predict(adaline.test_data)




