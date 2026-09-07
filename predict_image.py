import numpy as np
from model import NeuralNetwork
from PIL import Image



print("Trained model loaded successfully.")

def prepare_image(image_path):

    image = Image.open(image_path)

    image = image.convert("L")

    image_array = np.array(image)

    # Invert:
    # white background -> black
    # black digit -> white
    image_array = 255 - image_array

    # Remove weak background noise
    image_array[image_array < 30] = 0

    # Find the digit area
    rows = np.any(image_array > 0, axis=1)
    cols = np.any(image_array > 0, axis=0)

    row_indices = np.where(rows)[0]
    col_indices = np.where(cols)[0]

    if len(row_indices) == 0 or len(col_indices) == 0:
        return None

    top = row_indices[0]
    bottom = row_indices[-1]

    left = col_indices[0]
    right = col_indices[-1]

    # Crop digit
    digit = image_array[
        top:bottom + 1,
        left:right + 1
    ]

    digit = Image.fromarray(
        digit.astype(np.uint8)
    )

    # Keep the aspect ratio
    digit.thumbnail((20, 20))

    # Create MNIST-sized black canvas
    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )

    # Initial centering
    x = (28 - digit.width) // 2
    y = (28 - digit.height) // 2

    digit_array = np.array(digit)

    canvas[
        y:y + digit.height,
        x:x + digit.width
    ] = digit_array

    # -------------------------
    # CENTER OF MASS
    # -------------------------

    total = np.sum(canvas)

    if total > 0:

        rows_sum = np.sum(
            canvas,
            axis=1
        )

        cols_sum = np.sum(
            canvas,
            axis=0
        )

        center_y = np.sum(
            np.arange(28) * rows_sum
        ) / total

        center_x = np.sum(
            np.arange(28) * cols_sum
        ) / total

        shift_y = int(
            round(13.5 - center_y)
        )

        shift_x = int(
            round(13.5 - center_x)
        )

        canvas = np.roll(
            canvas,
            shift_y,
            axis=0
        )

        canvas = np.roll(
            canvas,
            shift_x,
            axis=1
        )

    # Normalize
    final_array = (
        canvas.astype(float)
        / 255.0
    )

    final_array = final_array.reshape(
        1,
        784
    )

    return final_array

model = NeuralNetwork()

correct = 0
total = 10

print("\nCUSTOM DIGIT TEST")
print("-----------------------------")

for actual_digit in range(10):

    image_path = f"my_digits/digit{actual_digit}.png"

    input_image = prepare_image(
        image_path
    )

    if input_image is None:

        print(
            actual_digit,
            "-> No digit detected"
        )

        continue

    predicted_digit, confidence, output = model.predict(
    input_image)

    if predicted_digit == actual_digit:

        correct += 1
        result = "Correct"

    else:

        result = "Wrong"

    print(
        "Actual:",
        actual_digit,
        "| Predicted:",
        predicted_digit,
        "| Confidence:",
        round(confidence, 2),
        "%",
        "|",
        result
    )
custom_accuracy = (
    correct / total
) * 100

print(
    "\nCustom Image Accuracy:",
    round(custom_accuracy, 2),
    "%"
)