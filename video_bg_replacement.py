import cv2
import numpy as np
from rembg import remove
from PIL import Image

# List of background images (Supports both JPG and PNG)
background_paths = ["img1.png", "img3.png", "input.jpg", "img4.png"]
current_bg_index = 0  # Default background index

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Load selected background
    background = cv2.imread(background_paths[current_bg_index])
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    background_resized = cv2.resize(background, (frame.shape[1], frame.shape[0]))

    # Convert frame to PIL image
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Remove background
    output_pil = remove(frame_pil)
    output_np = np.array(output_pil)

    # Ensure size matches frame
    output_np = cv2.resize(output_np, (frame.shape[1], frame.shape[0]))

    # Apply mask
    if output_np.shape[2] == 4:
        mask = output_np[:, :, 3] > 0
        final_frame = background_resized.copy()
        final_frame[mask] = output_np[:, :, :3][mask]
    else:
        final_frame = output_np[:, :, :3]

    final_frame = cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow("Video Background Replacement", final_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
        current_bg_index = int(chr(key)) - 1

cap.release()
cv2.destroyAllWindows()
