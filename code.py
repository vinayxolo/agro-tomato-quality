
from flask import Flask, render_template, request
import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import time
import serial_rx_tx
import webbrowser

# Disable eager execution for TensorFlow v1
tf.compat.v1.disable_eager_execution()

# Initialize Serial Communication
serialPort = serial_rx_tx.SerialPort()

# Categories for classification
CATEGORIES = ["Damaged", "Ripe", "Unripe"]

# Track last sent command to prevent continuous sending
last_sent_command = None

def OpenCommand():
    """Open serial port communication."""
    comport = 'COM5'
    baudrate = '9600'
    serialPort.Open(comport, baudrate)

def SendDataCommand(cmd):
    """Send a command to the serial port only once per new detection."""
    global last_sent_command
    message = str(cmd)
    if serialPort.IsOpen() and last_sent_command != message:
        serialPort.Send(message)
        last_sent_command = message  # Update last sent command

# Open Serial Communication
OpenCommand()

# Initialize Flask App
app = Flask(__name__)

@app.route('/')
def index():
    """Render the main index page."""
    return render_template("index.html", name="Project")

@app.route('/prediction', methods=["GET", "POST"])
def prediction():
    """Handle image prediction."""
    global last_sent_command

    # Ensure the test directory exists
    test_image_path = './data/test/test.jpg'
    if not os.path.exists('./data/train'):
        raise Exception("Error: Training directory does not exist.")
    if not os.path.exists('./data/test'):
        raise Exception("Error: Testing directory does not exist.")

    # Get uploaded image and save it
    img = request.files['img']
    img.save(test_image_path)

    # Image processing parameters
    image_size = 128
    num_channels = 3
    images = []

    if os.path.exists(test_image_path):
        # Read and preprocess image
        image = cv2.imread(test_image_path)
        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
        images.append(image)
        images = np.array(images, dtype=np.uint8).astype('float32') / 255.0
        x_batch = images.reshape(1, image_size, image_size, num_channels)

        # Load TensorFlow model
        sess = tf.Session()
        saver = tf.train.import_meta_graph('model/trained_model.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        
        graph = tf.get_default_graph()
        y_pred = graph.get_tensor_by_name("y_pred:0")
        x = graph.get_tensor_by_name("x:0")
        y_true = graph.get_tensor_by_name("y_true:0")

        # Placeholder for predictions
        y_test_images = np.zeros((1, len(os.listdir('./data/train'))))

        # Run prediction
        feed_dict_testing = {x: x_batch, y_true: y_test_images}
        result = sess.run(y_pred, feed_dict=feed_dict_testing)

        # Extract predicted class
        a = result[0].tolist()
        index1 = a.index(max(a))
        predicted_class = f"{CATEGORIES[index1]} Conf:{result[0][index1] * 100:.2f}%"

        # Command mapping for serial communication
        command_map = {"Damaged": "1", "Ripe": "2", "Unripe": "3"}
        detected_class = CATEGORIES[index1]

        # Send command based on prediction
        if detected_class in command_map:
            SendDataCommand(command_map[detected_class])

    else:
        predicted_class = "File does not exist"

    return render_template("prediction.html", data=predicted_class)

if __name__ == "__main__":
    webbrowser.open('http://127.0.0.1:5000/')
    app.run("127.0.0.1", port=5000, debug=False)
