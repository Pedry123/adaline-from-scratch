import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

args = {
    'lr': 0.0025,
    'min_error': 1e-6
}

def get_normalization_params(data):
    return {
        'min': data.min(),
        'max': data.max()
    }

def normalize(data, norm_params=None):
    if norm_params is None:
        norm_params = get_normalization_params(data)
    
    return (data - norm_params['min']) / (norm_params['max'] - norm_params['min'])

def pre_processing(df, train=True, norm_params=None):
    if train:
        outputs = df['d']
        outputs = np.array(outputs, dtype=float)
        inputs = df.drop('d', axis=1) # axis 1 é coluna
        inputs = np.array(df.drop('d', axis=1), dtype=float)
    else:
        inputs = np.array(df, dtype=float)
    inputs = normalize(inputs, norm_params)
    inputs = np.insert(inputs, 0, -1, axis=1).transpose() # adiciona o limiar e transpõe a matriz
    args['input_size'] = inputs.shape[0]
    args['num_samples'] = inputs.shape[1]
  
    return (inputs, outputs) if train else inputs

def mean_squared_error(desired_output, weights, inputs, num_samples):
    return np.sum((desired_output - np.dot(weights, inputs)) ** 2) / num_samples

def initialize_weights(input_size):
    weights = np.random.rand(input_size).reshape(1, input_size)
    return weights

def train(weights, inputs, outputs, lr, num_samples):
    epochs = 0
    while True:
        print('Initial weight vectors: ', weights)
        previous_eqm = mean_squared_error(outputs, weights, inputs, args['num_samples'])
        for i in range(num_samples):
            activation_potential = np.dot(weights, inputs[:, i]) # agora, vamos linha a linha 
            weights = weights + lr * (outputs[i] - activation_potential) * inputs[:, i]
        current_eqm = mean_squared_error(outputs, weights, inputs, args['num_samples'])
        epochs += 1
        print('Epoch:', epochs, 'Error:', current_eqm)
        if abs(previous_eqm - current_eqm) < args['min_error']:
            print('Final weight vectors: ', weights)
            print('Converged')
            break
    return weights

def test(weights, inputs):
    for i in range(args['num_samples']):
        activation_potential = np.dot(weights, inputs[:, i])
        print('Sample', i, 'Output:', 1 if activation_potential > 0 else -1)

if __name__ == "__main__":
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    norm_params = get_normalization_params(train_df.values)
    inputs, outputs = pre_processing(train_df)
    weights = initialize_weights(args['input_size'])
    weights = train(weights, inputs, outputs, args['lr'], args['num_samples'])
    test_inputs = pre_processing(test_df, train=False, norm_params=norm_params)
    test(weights, test_inputs)