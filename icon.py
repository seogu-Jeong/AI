import os
from PIL import Image, ImageDraw

def create_icon():
    # Configuration
    size = 1024
    bg_color = (0, 0, 128)  # Deep Navy #000080
    white = (255, 255, 255)
    dark_gray = (128, 128, 128)
    green = (0, 255, 0)
    
    # Create image
    img = Image.new('RGB', (size, size), bg_color)
    draw = ImageDraw.Draw(img)
    
    # 1. Raised border effect (Win98 style)
    # Top and Left: White
    draw.line([(0, 0), (size, 0)], fill=white, width=2)
    draw.line([(0, 0), (0, size)], fill=white, width=2)
    # Bottom and Right: Dark Gray
    draw.line([(0, size-1), (size, size-1)], fill=dark_gray, width=2)
    draw.line([(size-1, 0), (size-1, size)], fill=dark_gray, width=2)
    
    # 2. Large white upward-pointing arrow
    # Center: 512, 400 (roughly)
    # Arrow head (triangle)
    head_points = [
        (size // 2, 150),           # Top tip
        (size // 2 - 200, 400),     # Bottom left of head
        (size // 2 + 200, 400)      # Bottom right of head
    ]
    draw.polygon(head_points, fill=white)
    
    # Arrow stem (rectangle)
    stem_width = 120
    draw.rectangle([
        (size // 2 - stem_width // 2, 400),
        (size // 2 + stem_width // 2, 650)
    ], fill=white)
    
    # 3. 'SS' text below the arrow
    # Since we can't use external fonts and default is tiny, we'll draw 'SS' using shapes
    def draw_s(x_offset, y_offset, scale=1.0):
        # Drawing an 'S' shape using a thick line
        w, h = 140 * scale, 200 * scale
        t = 40 * scale # thickness
        # Points for 'S'
        points = [
            (x_offset + w, y_offset),               # Top right
            (x_offset, y_offset),                   # Top left
            (x_offset, y_offset + h//2),            # Middle left
            (x_offset + w, y_offset + h//2),        # Middle right
            (x_offset + w, y_offset + h),           # Bottom right
            (x_offset, y_offset + h)                # Bottom left
        ]
        draw.line(points, fill=white, width=int(t), joint="curve")

    # Draw two 'S's
    draw_s(320, 720)
    draw_s(560, 720)
    
    # 4. Small green dot in top-right corner
    dot_radius = 40
    dot_margin = 80
    draw.ellipse([
        (size - dot_margin - dot_radius, dot_margin - dot_radius),
        (size - dot_margin + dot_radius, dot_margin + dot_radius)
    ], fill=green)
    
    # Save the file
    output_path = os.path.join(os.path.dirname(__file__), 'icon.png')
    img.save(output_path)
    print(f"Icon saved to {output_path}")

if __name__ == "__main__":
    create_icon()
