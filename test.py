import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import joblib
import os
import sys
from datetime import timedelta


def load_model_and_classes():
    """Load the trained model and class names"""
    model_path = 'activity_recognition_model.keras'
    
    if not os.path.exists(model_path):
        print("Error: Model file 'activity_recognition_model.keras' not found!")
        print("Please train the model first using train.ipynb")
        sys.exit(1)
    
    class_names_path = 'class_names.pkl'
    
    if not os.path.exists(class_names_path):
        print(f"Error: Class names file '{class_names_path}' not found!")
        print("Please train the model first using train.ipynb")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path)
    class_names = joblib.load(class_names_path)
    
    print(f"Model loaded successfully!")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Classes: {class_names}")
    return model, class_names


def preprocess_frame(frame):
    """Preprocess a single frame for prediction:
    - Resize to (160, 160)
    - Convert BGR to RGB
    - Preprocess with EfficientNet preprocess_input function
    """
    frame_resized = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_LINEAR)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_rgb = frame_rgb.astype(np.float32)
    frame_array = np.expand_dims(frame_rgb, axis=0)
    frame_preprocessed = preprocess_input(frame_array)
    return frame_preprocessed


def predict_activity(model, frame, class_names):
    """Predict activity from a frame"""
    frame_preprocessed = preprocess_frame(frame)
    predictions = model(frame_preprocessed, training=False)
    if hasattr(predictions, 'numpy'):
        predictions = predictions.numpy()
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    activity_name = class_names[predicted_class_idx]
    return activity_name, confidence


def draw_prediction(frame, activity, confidence, frame_number=None, threshold=0.3):
    """Draw prediction on frame with threshold filtering"""
    if confidence < threshold:
        activity = "Unknown"
        color = (0, 0, 255)
    else:
        color = (0, 255, 0)
    
    if frame_number is not None:
        text = f"Frame {frame_number}: {activity} ({confidence:.2%})"
    else:
        text = f"{activity}: {confidence:.2%}"
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    cv2.rectangle(frame, (10, 10), (10 + text_width + 10, 10 + text_height + 10), (0, 0, 0), -1)
    cv2.putText(frame, text, (15, 10 + text_height), font, font_scale, color, thickness)
    return frame


def format_timestamp(frame_number, fps):
    """Convert frame number and fps to timestamp string 'MM:SS'"""
    total_seconds = frame_number / fps
    td = timedelta(seconds=total_seconds)
    minutes = int(td.seconds // 60)
    seconds = int(td.seconds % 60)
    return f"{minutes:02d}:{seconds:02d}"


def process_video_file(model, class_names, video_path, threshold=0.3):
    """Process video file frame by frame at slow frame rate and save results with timestamp"""
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' not found!")
        return
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {width}x{height}, {original_fps:.2f} FPS, {total_frames} frames")
    print(f"Confidence threshold: {threshold:.1%} (predictions below this will show as 'Unknown')")
    print("Press 'q' to quit")
    print("Processing each frame slowly and saving results to result.txt...")
    
    result_file = open('result.txt', 'w', encoding='utf-8')
    result_file.write("Frame Number | Predicted Activity | Timestamp\n")
    result_file.write("-" * 50 + "\n")
    
    frame_count = 0
    
    # Slow playback to roughly 2 FPS for better prediction (modify as needed)
    slow_fps = 2
    display_delay = int(1000 / slow_fps)  # ms delay between frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video")
            break
        
        activity, confidence = predict_activity(model, frame, class_names)
        if confidence < threshold:
            activity_display = "Unknown"
        else:
            activity_display = activity
        
        timestamp = format_timestamp(frame_count, original_fps)
        
        # Write result with frame number, predicted label, and timestamp
        result_file.write(f"frame {frame_count} predicted {activity_display} timestamp {timestamp} sec\n")
        
        # Draw prediction on the frame
        frame = draw_prediction(frame, activity_display, confidence, frame_number=frame_count, threshold=threshold)
        
        cv2.imshow('Human Activity Recognition - Video', frame)
        
        key = cv2.waitKey(display_delay) & 0xFF
        if key == ord('q'):
            print("Quitting video processing")
            break
        
        frame_count += 1
    
    cap.release()
    cv2.destroyAllWindows()
    result_file.close()
    print(f"\nVideo processing completed!")
    print(f"Total frames processed: {frame_count}")
    print(f"Results saved to: result.txt")


def main():
    """Main function only supports video file input for human activity recognition"""
    print("=" * 60)
    print("Human Activity Recognition - Testing (Video File Only)")
    print("=" * 60)
    
    # Load model and classes
    model, class_names = load_model_and_classes()
    
    threshold = 0.3
    print(f"\nConfidence threshold set to: {threshold:.1%}")
    print("(Predictions below this threshold will be marked as 'Unknown')")
    
    video_path = input("Enter video file path: ").strip()
    video_path = video_path.strip('"').strip("'")
    
    process_video_file(model, class_names, video_path, threshold=threshold)


if __name__ == "__main__":
    main()
