from PIL import Image, ImageDraw, ImageFont
import random
import os
import string
import math

font_dir = "../fonts/"
fonts = [os.path.join(font_dir, f) for f in os.listdir(font_dir) if f.endswith((".ttf", ".otf"))]

def generate_easy_image(text, save_path):
    # print('koo')
    width = 400
    height = 100
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_size = 60
    font = ImageFont.truetype('../fonts/Roboto.ttf', font_size)

    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    x = (width - text_width) // 2
    y = (height - text_height) // 2

    draw.text((x, y), text, fill=(0, 0, 0), font=font)
    img.save(save_path)
    # print('saved at', save_path)

def generate_hard_image(text, save_path):
    width = 400
    height = 100
    img = Image.new('RGB', (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    font_path = random.choice(fonts)
    font_size = random.randint(45, 65)
    font = ImageFont.truetype(font_path, font_size)

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

    img.save(save_path)
    return text

def generate_bonus_image(text, save_path, background_color="green"):
    width = 400
    height = 100
    
    if background_color == "green":
        bg_color = (100, 255, 100)
        text_to_render = text
    elif background_color == "red":
        bg_color = (255, 100, 100)
        text_to_render = text[::-1]
    else:
        bg_color = (255, 255, 255)
        text_to_render = text

    img = Image.new('RGB', (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    font_path = random.choice(fonts)
    font_size = random.randint(45, 65)
    font = ImageFont.truetype(font_path, font_size)

    total_width = 0
    for char in text_to_render:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        total_width += char_width

    x_offset = (width - total_width) // 2
    baseline_y = height // 2

    for char in text_to_render:
        bbox = draw.textbbox((0, 0), char, font=font)
        char_width = bbox[2] - bbox[0]
        char_height = bbox[3] - bbox[1]
        
        y_offset = baseline_y - char_height//2 + random.randint(-5, 5)
        
        draw.text((x_offset, y_offset), char, 
                 fill=(0, 0, 0), font=font)
        x_offset += char_width + random.randint(1, 3) 

    for _ in range(4):
        points = []
        x = 0
        amplitude = random.randint(5, 15)
        frequency = random.uniform(0.02, 0.04)
        while x < width:
            y = height//2 + amplitude * math.sin(frequency * x)
            points.append((x, y))
            x += 5
        
        if len(points) > 1:
            draw.line(points, fill=(random.randint(200, 240), 
                                  random.randint(200, 240), 
                                  random.randint(200, 240)), 
                     width=random.randint(1, 2))

    img.save(save_path)
    return text

import random
import string

letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
def generate_random_word():
    length = random.randint(4, 7)
    word = ''.join(random.choices(letters, k=length))
    return word

def create_dataset(size):
    for i in range(size):
        text = generate_random_word()
        #print(text)
        easy_text = text[:1].upper() + text[1:].lower()
        generate_easy_image(easy_text, save_path=f'easy/{easy_text}.png')
        text = generate_random_word()
        generate_hard_image(text, save_path=f'hard/{text}.png')
        text = generate_random_word()
        if i%2==0:
            generate_bonus_image(text, save_path = f'bonus/{text}.png', background_color='green')
        else:
            text = text[::-1]
            generate_bonus_image(text, save_path = f'bonus/{text}.png', background_color='red')
        
    

create_dataset(1000)