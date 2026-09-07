import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
model = np.load("mnist_model.npz")

W1 = model["W1"]
b1 = model["b1"]

W2 = model["W2"]
b2 = model["b2"]

print("Trained model loaded successfully.")
def relu(x):

    return np.maximum(0, x)
def softmax(x):

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
def forward(X):

    Z1 = np.dot(X, W1) + b1

    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2

    A2 = softmax(Z2)

    return A2
def prepare_image(image_path):

    # Open image
    image = Image.open(image_path)

    # Convert to grayscale
    image = image.convert("L")

    # Convert to numpy array
    image_array = np.array(image)

    # Invert image:
    # white background -> black
    # black digit -> white
    image_array = 255 - image_array

    # Find pixels belonging to the digit
    rows = np.any(image_array > 30, axis=1)
    cols = np.any(image_array > 30, axis=0)

    # Find boundaries of digit
    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        print("No digit detected in image.")
        return None

    top = row_indices[0]
    bottom = row_indices[-1]

    left = col_indices[0]
    right = col_indices[-1]

    # Crop only the digit
    digit = image_array[
        top:bottom + 1,
        left:right + 1
    ]

    # Convert back to PIL image
    digit = Image.fromarray(
        digit.astype(np.uint8)
    )

    # Resize digit to fit inside 20 x 20
    digit.thumbnail((20, 20))

    # Create empty 28 x 28 black image
    final_image = Image.new(
        "L",
        (28, 28),
        0
    )

    # Calculate position to center digit
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    # Place digit in center
    final_image.paste(
        digit,
        (x, y)
    )

    # Convert to numpy
    final_array = np.array(final_image)

    # Normalize
    final_array = final_array / 255.0

    # Flatten into 784 values
    final_array = final_array.reshape(1, 784)

    return final_array
image_path = "my_digits/digit5.png"
input_image = prepare_image(image_path)

print("Input image shape:", input_image.shape)
output = forward(input_image)

predicted_digit = np.argmax(output)
print("Predicted Digit:", predicted_digit)

print("\nPrediction probabilities:")

for digit in range(10):

    print(
        digit,
        ":",
        round(output[0][digit] * 100, 2),
        "%"
    )

display_image = input_image.reshape(28, 28)

plt.imshow(
    display_image,
    cmap="gray"
)

plt.title(
    f"Predicted Digit: {predicted_digit}"
)

plt.show()
