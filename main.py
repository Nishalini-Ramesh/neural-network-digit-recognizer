import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_openml


print("Loading MNIST dataset...")

mnist = fetch_openml("mnist_784", version=1, as_frame=False)

X = mnist.data
y = mnist.target.astype(int)

print("Dataset loaded successfully")

print("Image data shape:", X.shape)
print("Label data shape:", y.shape)

image = X[0].reshape(28, 28)

X = X / 255.0
X_train = X[:60000]
y_train = y[:60000]

X_test = X[60000:]
y_test = y[60000:]


print(X_train.shape)
print(X_test.shape)
input_size = 784
hidden_size = 64
output_size = 10

W1 = np.random.randn(
    input_size,
    hidden_size
) * np.sqrt(2 / input_size)

b1 = np.zeros(
    (1, hidden_size)
)

W2 = np.random.randn(
    hidden_size,
    output_size
) * np.sqrt(2 / hidden_size)

b2 = np.zeros(
    (1, output_size)
)

print("W1:", W1.shape)
print("b1:", b1.shape)

print("W2:", W2.shape)
print("b2:", b2.shape)

def relu(x):
    return np.maximum(0, x)

def softmax(x):

    x = x - np.max(x, axis=1, keepdims=True)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def forward(X):

    Z1 = np.dot(X, W1) + b1

    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2

    A2 = softmax(Z2)

    return Z1, A1, Z2, A2

def one_hot(y):

    one_hot_y = np.zeros((y.size, 10))

    one_hot_y[np.arange(y.size), y] = 1

    return one_hot_y
def relu_derivative(x):

    return x > 0

def backward(X, y, Z1, A1, A2):

    m = X.shape[0]

    one_hot_y = one_hot(y)

    dZ2 = A2 - one_hot_y

    dW2 = np.dot(A1.T, dZ2) / m

    db2 = np.sum(dZ2, axis=0, keepdims=True) / m


    dZ1 = np.dot(dZ2, W2.T) * relu_derivative(Z1)

    dW1 = np.dot(X.T, dZ1) / m

    db1 = np.sum(dZ1, axis=0, keepdims=True) / m


    return dW1, db1, dW2, db2

def update_parameters(dW1, db1, dW2, db2, learning_rate):

    global W1, b1, W2, b2

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1

    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
def get_predictions(A2):

    return np.argmax(A2, axis=1)
def get_accuracy(predictions, y):

    return np.mean(predictions == y)
def train(X, y, epochs, learning_rate):

    for epoch in range(epochs):

        Z1, A1, Z2, A2 = forward(X)

        dW1, db1, dW2, db2 = backward(
            X,
            y,
            Z1,
            A1,
            A2
        )

        update_parameters(
            dW1,
            db1,
            dW2,
            db2,
            learning_rate
        )

        if epoch % 10 == 0:

            predictions = get_predictions(A2)

            accuracy = get_accuracy(
                predictions,
                y
            )

            print(
                    "Epoch:",
                    epoch,
                    "Accuracy:",
                    round(accuracy * 100, 2),"%"
)

print("Training started...")

train(
    X_train,
    y_train,
    epochs=150,
    learning_rate=0.1
)

print("Training completed.")


_, _, _, test_output = forward(X_test)

test_predictions = get_predictions(test_output)

test_accuracy = get_accuracy(
    test_predictions,
    y_test
)

print(
    "Test Accuracy:",
    round(test_accuracy * 100, 2),
    "%"
)

np.savez(
    "mnist_model.npz",
    W1=W1,
    b1=b1,
    W2=W2,
    b2=b2
)

print("Model saved successfully.")

index = 10

image = X_test[index]

actual_digit = y_test[index]

_, _, _, output = forward(
    image.reshape(1, -1)
)

predicted_digit = get_predictions(output)[0]


print(
    "Actual Digit:",
    actual_digit
)

print(
    "Predicted Digit:",
    predicted_digit
)


plt.imshow(
    image.reshape(28, 28),
    cmap="gray"
)

plt.title(
    f"Actual: {actual_digit}, Predicted: {predicted_digit}"
)

plt.show()






