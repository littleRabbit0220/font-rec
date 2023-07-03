import os
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import json

from flask import Flask, request
from flask_cors import CORS
# from imutils import paths

from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

BATCH_SIZE = 32

IMG_SIZE = 224

# DATA_PATH = 'D:\\Downloads\\font-images'
# images = []
# labels = []
# image_paths = sorted(list(paths.list_images(DATA_PATH)))

# for image_path in image_paths:
#     label = image_path.split(os.path.sep)[-2]
#     labels.append(label)
#     images.append(image_path)

# unique_labels = np.unique(labels)

# with open('output.txt', "w") as file:
#     file.write("', '".join(unique_labels))

with open('data.json', "r") as file:
    unique_labels = json.load(file)

print(unique_labels)

def process_image(image_path):
    """
    Takes an image file path and turns it into a Tensor.
    """
    # Read in image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired size (224, 244)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
    """
    Creates batches of data out of image (x) and label (y) pairs.
    Shuffles the data if it's training data but doesn't shuffle it if it's validation data.
    Also accepts test data as input (no labels).
    """
    # If the data is a test dataset, we probably don't have labels
    if test_data:
        print("Creating test data batches...")
        data = tf.data.Dataset.from_tensor_slices((tf.constant(x))) # only filepaths
        data_batch = data.map(process_image).batch(BATCH_SIZE)
        return data_batch

@app.route('/api/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return {"state": "error", "message": "No file part"}
    file = request.files['file']
    if file.filename == '':
        return {"state": "error", "message": "No selected file"}
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    test_data = create_data_batches([filepath], test_data=True)

    model = tf.keras.models.load_model(os.path.join('top_model.h5'), custom_objects={"KerasLayer":hub.KerasLayer})
    predictions = model.predict(test_data, verbose=1)
    print(predictions[0])
    print(f"Max value (probability of prediction): {np.max(predictions[0])}") # the max probability value predicted by the model
    print(f"Sum: {np.sum(predictions[0])}") # because we used softmax activation in our model, this will be close to 1
    print(f"Max index: {np.argmax(predictions[0])}") # the index of where the max value in predictions[0] occurs
    print(f"Predicted label: {unique_labels[np.argmax(predictions[0])]}") # the predicted label

    for prediction in predictions:
        print(np.max(prediction), unique_labels[np.argmax(prediction)])

    return unique_labels[np.argmax(predictions[0])].split('-')[0]

if __name__ == '__main__':
    app.run()