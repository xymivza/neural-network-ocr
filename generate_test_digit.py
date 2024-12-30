from PIL import Image, ImageDraw, ImageFont
import os

def create_digit_image(digit, size=(28, 28), filename="test_digit.png"):
    image = Image.new('L', size, 'white')
    draw = ImageDraw.Draw(image)
    
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    w, h = draw.textsize(str(digit), font=font)
    draw.text(
        ((size[0]-w)/2, (size[1]-h)/2),
        str(digit),
        fill='black',
        font=font
    )
    
    image.save(filename)
    print(f"Created test image: {filename}")

if __name__ == "__main__":
    os.makedirs("test_images", exist_ok=True)
    
    for digit in range(10):
        create_digit_image(digit, filename=f"test_images/digit_{digit}.png")