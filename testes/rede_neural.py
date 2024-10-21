import numpy as np

# Função de ativação sigmoid
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada da função sigmoid
def sigmoid_derivative(x):
    return x * (1 - x)

# Inicialização dos parâmetros
def initialize_parameters(input_size, hidden_size, output_size):
    W1 = np.random.rand(hidden_size, input_size) - 0.5
    W2 = np.random.rand(output_size, hidden_size) - 0.5
    return W1, W2

# Propagação direta
def forward_propagation(X, W1, W2):
    Z1 = np.dot(W1, X)
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1)
    A2 = sigmoid(Z2)
    return Z1, A1, Z2, A2

# Retropropagação
def backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2, learning_rate):
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T)
    dZ1 = np.dot(W2.T, dZ2) * sigmoid_derivative(A1)
    dW1 = np.dot(dZ1, X.T)

    # Atualizar os pesos
    W1 -= learning_rate * dW1
    W2 -= learning_rate * dW2
    return W1, W2

# Treinamento
def train(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate):
    W1, W2 = initialize_parameters(input_size, hidden_size, output_size)
    
    for epoch in range(epochs):
        for i in range(len(X_train)):
            X = X_train[i].reshape(-1, 1)  # Dados de entrada
            Y = Y_train[i].reshape(-1, 1)  # Saída esperada

            # Propagação
            Z1, A1, Z2, A2 = forward_propagation(X, W1, W2)

            # Retropropagação
            W1, W2 = backpropagation(X, Y, Z1, A1, Z2, A2, W1, W2, learning_rate)

        if epoch % 100 == 0:
            loss = np.mean((A2 - Y_train)**2)
            print(f"Epoch {epoch}, Loss: {loss}")
    
    return W1, W2

# Dados de exemplo - simplificados (apenas 4 exemplos, mas pode usar o dataset MNIST real)
X_train = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 0, 0, 1], [1, 1, 1, 1]])
Y_train = np.array([[0], [1], [1], [0]])

input_size = X_train.shape[1]
hidden_size = 2
output_size = 1
epochs = 1000
learning_rate = 0.1

# Treinar a rede neural
train(X_train, Y_train, input_size, hidden_size, output_size, epochs, learning_rate)
