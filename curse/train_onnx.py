import cv2
import numpy as np
import onnxruntime as ort

def load_model(model_path):
    # Load the ONNX model
    session = ort.InferenceSession(model_path)
    return session

def preprocess_image(image, input_size):
    # Resize and normalize the image
    image_resized = cv2.resize(image, (input_size, input_size))
    image_normalized = image_resized.astype(np.float32) / 255.0  # Normalize to [0, 1]
    image_transposed = np.transpose(image_normalized, (2, 0, 1))  # Change from HWC to CHW
    return image_transposed[np.newaxis, :]  # Add batch dimension

def post_process(output_data, image_shape, conf_threshold=0.5):
    h, w = image_shape
    detections = []

    # Check if output_data is a list and extract the first element if so
    if isinstance(output_data, list):
        output_data = output_data[0]  # Assuming the first element contains the relevant output

    # Ensure output_data is a NumPy array
    if isinstance(output_data, np.ndarray):
        output_data = output_data  # It's already a NumPy array
    else:
        print("Error: output_data is not a NumPy array.")

    # Iterate through each detection
    for i in range(output_data.shape[1]):
        confidence = output_data[0, i, 4]  # Confidence is usually the 5th value
        if confidence > conf_threshold:
            x_center = output_data[0, i, 0] * w
            y_center = output_data[0, i, 1] * h
            width = output_data[0, i, 2] * w
            height = output_data[0, i, 3] * h

            # Convert to top-left corner coordinates
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)

            detections.append((x, y, int(width), int(height), confidence))
            print(f"Detected object at: (x: {x}, y: {y}, width: {int(width)}, height: {int(height)}, confidence: {confidence:.2f})")  # Print coordinates
    return detections

def draw_detections(image, detections):
    for (x, y, width, height, confidence) in detections:
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f'Conf: {confidence:.2f}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main():
    model_path = "best_1.onnx"  # Your model's path
    session = load_model(model_path)

    # Load image
    image = cv2.imread("./input.jpg")
    if image is None:
        print("Error: Could not read image.")
        return

    input_size = 1920  # Adjust according to your model's input size
    input_tensor = preprocess_image(image, input_size)

    # Run inference
    output_data = session.run(None, {"images": input_tensor})  # Adjust input name if necessary

    # Debugging: Print the type and shape of output_data
    print(f"Output data type: {type(output_data)}")
    if isinstance(output_data, list):
        print(f"Output data length: {len(output_data)}")
        print(f"First element shape: {output_data[0].shape if isinstance(output_data[0], np.ndarray) else 'N/A'}")
    else:
        print(f"Output data shape: {output_data.shape if isinstance(output_data, np.ndarray) else 'N/A'}")

    # Post-process output
    detections = post_process(output_data, image.shape[:2])

    # Draw detections
    draw_detections(image, detections)

    # Show results
    cv2.imshow("Detections", image)
    cv2.imwrite("output.jpg", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()