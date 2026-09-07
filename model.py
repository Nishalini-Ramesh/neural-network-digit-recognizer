import numpy as np


class NeuralNetwork:

    def __init__(self, model_path="mnist_model.npz"):

        model = np.load(model_path)

        self.W1 = model["W1"]
        self.b1 = model["b1"]
        self.W2 = model["W2"]
        self.b2 = model["b2"]

        print("Trained model loaded successfully.")


    def relu(self, x):

        return np.maximum(0, x)


    def softmax(self, x):

        x = x - np.max(
            x,
            axis=1,
            keepdims=True
        )

        exp_x = np.exp(x)

        return exp_x / np.sum(
            exp_x,
            axis=1,
            keepdims=True
        )


    def forward(self, X):

        Z1 = np.dot(
            X,
            self.W1
        ) + self.b1

        A1 = self.relu(Z1)

        Z2 = np.dot(
            A1,
            self.W2
        ) + self.b2

        A2 = self.softmax(Z2)

        return A2


    def predict(self, X):

        output = self.forward(X)

        predicted_digit = np.argmax(
            output,
            axis=1
        )[0]

        confidence = np.max(
            output
        ) * 100

        return predicted_digit, confidence, output

if __name__ == "__main__":

    model = NeuralNetwork()

    print("Model is ready.")