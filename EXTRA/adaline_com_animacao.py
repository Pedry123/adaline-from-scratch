import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler

def plotline(data, weights, title):
    plt.scatter(data[:, 0], data[:, 1], c=data[:, -1], edgecolors='k')  # Corrigido para usar a última coluna (saída)

    xmin, xmax = plt.gca().get_xlim()
    ymin, ymax = plt.gca().get_ylim()

    x = np.linspace(-2, 2, 100)
    y = (-weights[1] * x - weights[0]) / weights[2]
    plt.plot(x, y, label=title)

    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)

def normalize(data, min, max):
   return (data - min) / (max - min)

def animate(i, data, weights_history, title):
    plt.clf()
    current_weights = weights_history[i]
    if i == 0:
        epoch_label = "- Época Inicial"
    elif i == len(weights_history) - 1:
        epoch_label = f'- Época Final'
    else:
        epoch_label = f'- Em Treinamento'
    plotline(data, current_weights, f'{title} {epoch_label}')
    return plt.gca().lines

def get_norm_params(data):
    norm_params = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return norm_params

def split_input_output(data, norm_params=True):
    inputs = data[:, :-1]
    outputs = data[:, -1]
    if norm_params:
        norm_params = get_norm_params(inputs)
    return (inputs, outputs, norm_params) if norm_params else (inputs, outputs)

def pre_processing(inputs, norm_params):
    #inputs = normalize(inputs, norm_params['min'], norm_params['max'])
    inputs = np.insert(inputs, 0, -1, axis=1).transpose()
    return inputs

def initialize_weights(input_size):
    weights = np.random.rand(input_size)
    return [0.49703716 ,0.75576251, 0.60571322]
    #return weights

def mean_squared_error(num_samples, outputs, weights, inputs):
    squared_error = 0
    for k in range(num_samples):
        prediction = np.dot(weights, inputs[:, k])
        squared_error += (outputs[k] - prediction) ** 2
    return squared_error / num_samples

class Adaline:
    def __init__(self, learning_rate, precision):
        self.learning_rate = learning_rate
        self.precision = precision
        self.errors_list = []
        self.weights_history = []


    def train(self, inputs, outputs, data, weights, num_samples):
        epoch = 0
        num_samples = inputs.shape[1]
        self.weights_history = [weights.copy()]
        print("Vetor de pesos inicial: ", weights)
        while True:
            previous_error = mean_squared_error(num_samples, outputs, weights, inputs)
            for k in range(num_samples):
                prediction = np.dot(weights, inputs[:, k])
                weights = weights + self.learning_rate * (outputs[k] - prediction) * inputs[:, k]
            self.weights_history.append(weights.copy())
            current_error = mean_squared_error(num_samples, outputs, weights, inputs)
            self.errors_list.append(current_error)
            epoch += 1
            print(f'Época: {epoch}, Erro: {current_error}, Pesos: {weights.reshape(-1)}')
            if abs(previous_error - current_error) < self.precision:
                break
        print("Vetor de pesos final: ", weights)
        return weights


    def predict(self, weights, inputs):
        pred =  np.dot(weights, inputs)
        return np.where(pred >= 0, 1, -1)


if __name__ == "__main__":
    #data = np.random.uniform(-1, 1, (40, 2))
    #test = np.random.uniform(-1, 1, (20, 2))
    #data = np.insert(data, 2, np.random.choice([-1, 1], 40), axis=1)
    #test = np.insert(test, 2, np.random.choice([-1, 1], 20), axis=1)
    data, outputs = make_blobs(n_samples=40, centers=2, n_features=2, random_state=167) #67, 671
    outputs = np.where(outputs == 0, -1, 1)
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data = np.insert(data, 2, outputs, axis=1)
    inputs, outputs, norm_params = split_input_output(data)
    
    inputs = pre_processing(inputs, norm_params)
    weights = initialize_weights(inputs.shape[0])

    adaline = Adaline(0.001, 1e-6)
    weights = adaline.train(inputs, outputs, data, weights, inputs.shape[1])

    # Criar animação
    fig = plt.figure(figsize=(10, 6))
    ani = FuncAnimation(fig, animate, frames=range(0, len(adaline.weights_history), 10),
                        fargs=(data, adaline.weights_history, 'ADALine'),
                        interval=10, blit=False, repeat=False)
    ani.save('adaline.gif', writer='imagemagick', fps=1)