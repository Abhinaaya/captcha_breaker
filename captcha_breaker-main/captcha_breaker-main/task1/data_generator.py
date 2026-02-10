from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import random
import os
import string
import math

font_dir = "../fonts/"
fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith((".ttf", ".otf", ".TTF"))]

def generate_easy_image(text, font, save_path):
    # print('koo')
    width = 400
    height = 100
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_size = 60
    font = ImageFont.truetype(font, font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    img.save(save_path)
    # print('saved at', save_path)
    
def generate_hard_image(text, font, save_path):
    width = 400
    height = 100
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    
    font_size = random.randint(45, 65)
    font = ImageFont.truetype(font, font_size)

    total_width = 0
    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        total_width += char_width

    x_offset = (width - total_width) // 2
    baseline_y = height // 2

    for char in text:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        
        y_offset = baseline_y - char_height//2 + random.randint(-5, 5)
        
        draw.text((x_offset, y_offset), char, 
                 fill=(0, 0, random.randint(0, 255)), font=font)
        x_offset += char_width + random.randint(1, 3) 

    for _ in range(3):
        points = []
        x = 0
        while x < width:
            y = random.randint(10, height-10)
            points.append((x, y))
            x += random.randint(20, 40)
        
        if len(points) > 1:
            draw.line(points, fill=(random.randint(200, 240), 
                                  random.randint(200, 240), 
                                  random.randint(200, 240)), 
                     width=random.randint(1, 2))

    # Add more variations
    rotation_angle = random.uniform(-5, 5)  # Slight rotation
    img = img.rotate(rotation_angle, expand=False, fillcolor='white')
    
    # Add random background noise occasionally
    if random.random() < 0.3:
        for _ in range(random.randint(50, 200)):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            draw.point((x, y), fill=(random.randint(0, 255), 
                                    random.randint(0, 255), 
                                    random.randint(0, 255)))
    
    # Add random brightness/contrast variation
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(random.uniform(0.8, 1.2))
    
    img.save(save_path)
    return text

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
def generate_random_word():
    length = random.randint(4, 7)
    word = ''.join(random.choices(letters, k=length))
    return word

def generate_data():
    for i in range(100):
        text = generate_random_word()
        easy_text = text[:1].upper() + text[1:].lower()
        os.makedirs(f'./data/{easy_text}', exist_ok=True)
        count = 1
        
        # Generate more variations per font
        for font in fonts:
            # Original versions
            generate_easy_image(easy_text, font=font, save_path=f'./data/{easy_text}/{count}.png')
            count += 1
            generate_hard_image(text, font=font, save_path=f'./data/{easy_text}/{count}.png')
            count += 1
            
            # Additional variations with different cases
            variations = [
                text.upper(),
                text.lower(),
                text.title()
            ]
            
            for var in variations:
                generate_hard_image(var, font=font, save_path=f'./data/{easy_text}/{count}.png')
                count += 1
    
generate_data()
        