import cv2
import time
import numpy as np
from collections import deque

# Binary Search Tree Node for storing motion event timestamps
class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

# BST class to store and manage motion timestamps
class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if self.root is None:
            self.root = BSTNode(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key)
            else:
                self._insert(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = BSTNode(key)
            else:
                self._insert(node.right, key)

    def inorder(self, node=None, result=None):
        if result is None:
            result = []
        if node is None:
            node = self.root
        if node.left:
            self.inorder(node.left, result)
        result.append(node.key)
        if node.right:
            self.inorder(node.right, result)
        return result

# Queue to store recent motion events (FIFO)
class MotionEventQueue:
    def __init__(self, max_size=10):
        self.queue = deque(maxlen=max_size)

    def add_event(self, event):
        self.queue.append(event)

    def get_events(self):
        return list(self.queue)

# Initialize the camera and data structures
camera = cv2.VideoCapture(0)
motion_queue = MotionEventQueue(max_size=10)
motion_tree = BST()

time.sleep(2)

# Variables for motion detection
first_frame = None
motion_detected = False

print("Press 'q' to quit.")

while True:
    ret, frame = camera.read()
    if not ret:
        break

    # Convert frame to grayscale and apply Gaussian blur
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

    # Set the first frame as reference
    if first_frame is None:
        first_frame = gray_frame
        continue

    # Calculate frame delta
    frame_delta = cv2.absdiff(first_frame, gray_frame)
    thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False

    for contour in contours:
        if cv2.contourArea(contour) < 1000:
            continue
        motion_detected = True
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Log motion event
    if motion_detected:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        motion_queue.add_event(timestamp)
        motion_tree.insert(timestamp)
        cv2.putText(frame, f"Motion Detected: {timestamp}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the video feed
    cv2.imshow("Surveillance", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
camera.release()
cv2.destroyAllWindows()

# Print motion events stored in the queue
print("\nRecent Motion Events (Queue):")
for event in motion_queue.get_events():
    print(event)

# Print motion events sorted (Inorder Traversal of BST)
print("\nSorted Motion Events (BST):")
sorted_events = motion_tree.inorder()
for event in sorted_events:
    print(event)
