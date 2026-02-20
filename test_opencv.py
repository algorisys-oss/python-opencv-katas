import cv2
import os
import tempfile

def test_image_read_write():
    # 1. Define the path to your image
    # REPLACE THIS with the actual path to an image on your computer
    # Use raw strings (r"...") or forward slashes to avoid syntax errors
    image_path = r"C:\Users\Vishaka\Downloads\pikachu.jfif" 

    # Check if file exists first
    if not os.path.exists(image_path):
        print(f"Error: File not found at {image_path}")
        print("Please edit the 'image_path' variable in this script to point to a valid image file.")
        return

    # 2. Read the image using cv2.imread()
    print(f"Attempting to read image from: {image_path}")
    img = cv2.imread(image_path)

    # Check if image was loaded successfully
    if img is None:
        print("Error: cv2.imread() failed to load the image. The file might be corrupted or the format is not supported.")
        return
    else:
        print(f"Success! Image loaded. Dimensions: {img.shape}")

    # 3. Create a temporary file to save the image
    # This creates a temp file with a .jpg extension
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
        temp_output_path = temp_file.name

    # 4. Save the image using cv2.imwrite()
    print(f"Attempting to save image to temporary file: {temp_output_path}")
    success = cv2.imwrite(temp_output_path, img)

    if success:
        print(f"Success! Image saved to: {temp_output_path}")
    else:
        print("Error: Failed to save video to temporary file.")

if __name__ == "__main__":
    test_image_read_write()
