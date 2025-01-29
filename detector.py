import cv2
import torch
import pyautogui
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Define the ArrowDetector class
class ArrowDetector:
    def __init__(self, model_path, confidence_threshold=0.7):
        """
        Initialize the YOLOv5 model.
        
        Args:
            model_path (str): Path to the YOLOv5 model weights.
            confidence_threshold (float): Confidence threshold for detection.
        """
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        self.model.conf = confidence_threshold  # Set confidence threshold

    def detect_arrows(self, frame):
        """
        Perform arrow detection on a single frame.
        
        Args:
            frame (numpy.ndarray): The input video frame.
        
        Returns:
            List of detections with bounding boxes and labels.
        """
        results = self.model(frame)
        return results.xyxy[0].cpu().numpy()  # Format: [x1, y1, x2, y2, confidence, class_id]

# Define a function to map detected classes to directions
def map_class_to_direction(class_id):
    """
    Map the detected class ID to a direction.
    
    Args:
        class_id (int): Detected class ID (0=up, 1=down, 2=left, 3=right).
    
    Returns:
        str: Direction as 'up', 'down', 'left', or 'right'.
    """
    direction_map = {0: "up", 1: "down", 2: "left", 3: "right"}
    return direction_map.get(class_id, None)

# Define a function to simulate game input
def simulate_game_input(direction):
    """
    Simulate keyboard input for a given direction.
    
    Args:
        direction (str): The direction ('up', 'down', 'left', or 'right').
    """
    if direction:
        pyautogui.press(direction.capitalize())

# Define the main video processing loop
def process_video(model_path, confidence_threshold=0.7):
    """
    Process video frames and detect arrows, mapping to game inputs.
    
    Args:
        model_path (str): Path to the YOLOv5 model weights.
        confidence_threshold (float): Confidence threshold for detection.
    """
    # Initialize video capture and the detector
    cap = cv2.VideoCapture(0)  # Use the default webcam
    detector = ArrowDetector(model_path, confidence_threshold)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect arrows in the frame
        detections = detector.detect_arrows(frame)
        print(detections)  # Debug log

        for detection in detections:
            x1, y1, x2, y2, conf, class_id = detection[:6]

            # Filter by confidence
            if conf > confidence_threshold:
                # Map class ID to direction
                direction = map_class_to_direction(int(class_id))
                print(direction)

                # Simulate game input
                simulate_game_input(direction)

                # Draw bounding box and label on the frame
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{direction} ({conf:.2f})",
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # Display the processed frame
        cv2.imshow("Arrow Detection - Live Feed", frame)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the script
if __name__ == "__main__":
    MODEL_PATH = "best.pt"  # Path to the trained YOLOv5 model weights
    CONFIDENCE_THRESHOLD = 0.7  # Confidence threshold for detection
    process_video(MODEL_PATH, CONFIDENCE_THRESHOLD)