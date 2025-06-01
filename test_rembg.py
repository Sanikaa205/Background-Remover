from rembg import remove
from PIL import Image

# Input and output file paths
input_path = "C:\BG remover\photo.jpg"  # Replace with your actual image file
output_path = "output.png"

try:
    # Open the image
    image = Image.open(input_path)

    # Convert to RGBA mode (fix potential issues)
    image = image.convert("RGBA")

    # Remove the background
    output = remove(image)

    # Save the output image
    output.save(output_path, format="PNG")

    print(f"✅ Background removed successfully! Saved as {output_path}")

except Exception as e:
    print(f"❌ Error: {e}")
