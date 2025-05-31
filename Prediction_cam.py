import tensorflow.compat.v1 as tf
import numpy as np
import os
import cv2
import time

tf.disable_eager_execution()

# Define categories
CATEGORIES = ["cutting", "non cutting"]
print("Categories:", CATEGORIES)

# Initialize video capture
video = cv2.VideoCapture(0)
time.sleep(2)

# Path of training and testing images
train_path = './data/train'
test_path = './data/test'

if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise Exception("Required directories do not exist!")

# Set image properties
image_size = 128
num_channels = 3

# Load the model
sess = tf.Session()
saver = tf.train.import_meta_graph('model/trained_model.meta')
saver.restore(sess, tf.train.latest_checkpoint('./model/'))
graph = tf.get_default_graph()
y_pred = graph.get_tensor_by_name("y_pred:0")
x = graph.get_tensor_by_name("x:0")
y_true = graph.get_tensor_by_name("y_true:0")
y_test_images = np.zeros((1, len(CATEGORIES)))

# Counter for processed images
image_count = 0

print("Starting the loop. Press 'q' to capture and classify an image, or 'ESC' to exit.")

# Start an infinite loop
while True:
    grabbed, frame = video.read()
    if not grabbed:
        print("Failed to capture video frame.")
        break

    # Show the live video frame
    cv2.imshow("Live Feed", frame)

    # Save a test image and process it when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        image_count += 1
        test_image_path = os.path.join(test_path, f'test_{image_count}.jpg')
        cv2.imwrite(test_image_path, frame)
        cv2.waitKey(1)  # Give time for the write to complete

        # Preprocess the saved test image
        if os.path.exists(test_image_path):
            print(f"Processing Image #{image_count} - {test_image_path}")
            image = cv2.imread(test_image_path)
            image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
            image = image.astype('float32') / 255.0  # Normalize image
            x_batch = image.reshape(1, image_size, image_size, num_channels)

            # Run prediction
            feed_dict_testing = {x: x_batch, y_true: y_test_images}
            result = sess.run(y_pred, feed_dict=feed_dict_testing)

            # Parse result
            confidence = result[0].max()
            predicted_index = result[0].argmax()
            predicted_class = CATEGORIES[predicted_index]
            print(f"Image #{image_count}: Prediction: {predicted_class}, Confidence: {confidence * 100:.2f}%")
        else:
            print(f"Image #{image_count}: Test image could not be saved or found.")

    # Display the image count on the frame
    display_text = f"Images Processed: {image_count}"
    cv2.putText(frame, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Live Feed", frame)

    # Exit loop when 'ESC' is pressed
    if cv2.waitKey(1) & 0xFF == 27:  # Escape key
        print("Exiting...")
        break

# Release video and close windows
video.release()
cv2.destroyAllWindows()
