import cv2
import mediapipe as mp
import time
import svgwrite
import os
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform

# Check for Apple Silicon
IS_ARM_MAC = platform.system() == 'Darwin' and platform.machine().startswith('arm')

# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Gesture thresholds
PINCH_THRESHOLD = 0.05
ALL_FINGERS_THRESH_TIME = 2.0
THUMBS_UP_HOLD_TIME = 2.0

# UI constants
BUTTON_WIDTH = 130
BUTTON_HEIGHT = 60
BUTTON_MARGIN = 20
COLOR_BOX_SIZE = 80
COLOR_BOX_MARGIN = 15
SLIDER_THICKNESS = 20

# Define available colors
COLORS = [
    {'name': 'Black', 'bgr': (0, 0, 0), 'hex': '#000000'},
    {'name': 'Red',   'bgr': (17, 21, 139), 'hex': '#8b1511'},
    {'name': 'Green', 'bgr': (114, 139, 17), 'hex': '#118b72'},
    {'name': 'Blue',  'bgr': (111, 57, 15), 'hex': '#0f396f'}
]

# Thickness range
THICKNESS_MIN = 1
THICKNESS_MAX = 10

# Landmark indices
INDEX_TIP = 8
THUMB_TIP = 4
MIDDLE_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Font settings
FONT_PATH = "fonts/Bauhaus93.ttf"  # Change this to your desired font
FONT_LARGE_SIZE = 36
FONT_MEDIUM_SIZE = 32
FONT_SMALL_SIZE = 20

# Text size cache
text_size_cache = {}

# Initialize fonts and cache common text sizes
def initialize_fonts_and_cache():
    global FONT_LARGE, FONT_MEDIUM, FONT_SMALL, USE_PIL_FONTS, text_size_cache
    
    try:
        FONT_LARGE = ImageFont.truetype(FONT_PATH, FONT_LARGE_SIZE)
        FONT_MEDIUM = ImageFont.truetype(FONT_PATH, FONT_MEDIUM_SIZE)
        FONT_SMALL = ImageFont.truetype(FONT_PATH, FONT_SMALL_SIZE)
        print(f"Loaded font: {FONT_PATH}")
        USE_PIL_FONTS = True
    except IOError:
        print(f"Font {FONT_PATH} not found, using default font")
        FONT_LARGE = ImageFont.load_default()
        FONT_MEDIUM = ImageFont.load_default()
        FONT_SMALL = ImageFont.load_default()
        USE_PIL_FONTS = False
    
    # Pre-cache common text sizes
    common_texts = ['Clear', 'Save', 'Exit', 'Undo', 'DRAWING AREA', 'AirCanvas']
    for text in common_texts:
        for font in [FONT_SMALL, FONT_MEDIUM, FONT_LARGE]:
            cache_key = f"{text}_{font.size}"
            if cache_key not in text_size_cache:
                dummy_img = Image.new('RGB', (1, 1))
                draw = ImageDraw.Draw(dummy_img)
                text_size = draw.textbbox((0, 0), text, font=font)
                text_width = text_size[2] - text_size[0]
                text_height = text_size[3] - text_size[1]
                text_size_cache[cache_key] = (text_width, text_height)

# Get text size with caching
def get_text_size(text, font):
    cache_key = f"{text}_{font.size}"
    if cache_key in text_size_cache:
        return text_size_cache[cache_key]
    
    # Create a small dummy image just for size calculation
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    text_size = draw.textbbox((0, 0), text, font=font)
    text_width = text_size[2] - text_size[0]
    text_height = text_size[3] - text_size[1]
    
    text_size_cache[cache_key] = (text_width, text_height)
    return text_width, text_height

# ARM Mac optimized matrix operations using numpy vectorization
def norm_dist_vectorized(a, b):
    return np.sqrt((a.x - b.x)**2 + (a.y - b.y)**2)

# Gesture detectors
def is_drawing_gesture(hand):
    if IS_ARM_MAC:
        return norm_dist_vectorized(hand.landmark[THUMB_TIP], hand.landmark[INDEX_TIP]) < PINCH_THRESHOLD
    else:
        return ((hand.landmark[THUMB_TIP].x - hand.landmark[INDEX_TIP].x)**2 + 
                (hand.landmark[THUMB_TIP].y - hand.landmark[INDEX_TIP].y)**2)**0.5 < PINCH_THRESHOLD

# Optimized PIL text rendering - uses ROI (Region of Interest)
def put_text_pil(img, text, position, font, text_color=(255, 255, 255)):
    # Get text dimensions
    text_width, text_height = get_text_size(text, font)
    
    # Add padding
    text_width += 20
    text_height += 20
    
    # Extract region of interest
    x, y = position
    
    # Ensure coordinates are within image bounds
    x_end = min(x + text_width, img.shape[1])
    y_end = min(y + text_height, img.shape[0])
    x = max(0, x)
    y = max(0, y)
    
    # If region is invalid, revert to OpenCV text
    if x >= x_end or y >= y_end:
        cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2, cv2.LINE_AA)
        return img
    
    # Extract region for text
    roi = img[y:y_end, x:x_end].copy()
    
    # Convert ROI to PIL Image
    pil_roi = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_roi)
    
    # Draw text with specified font (at 0,0 since this is a small ROI)
    color_rgb = (text_color[2], text_color[1], text_color[0])
    draw.text((0, 0), text, font=font, fill=color_rgb)
    
    # Convert back to OpenCV BGR format and place back in the image
    roi_result = cv2.cvtColor(np.array(pil_roi), cv2.COLOR_RGB2BGR)
    img[y:y_end, x:x_end] = roi_result
    
    return img

# Draw text with either PIL (for custom fonts) or OpenCV (fallback)
def draw_text(frame, text, position, font, text_color=(255, 255, 255)):
    if USE_PIL_FONTS:
        return put_text_pil(frame, text, position, font, text_color)
    else:
        cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 
                   0.8 if font == FONT_SMALL else 1.0 if font == FONT_MEDIUM else 1.2, 
                   text_color, 2, cv2.LINE_AA)
        return frame

# Helper: draw rounded, semi-transparent button with centered text
# Optimized to avoid unnecessary operations
def draw_button(frame, x1, y1, x2, y2, color, text='', alpha=0.8, radius=10):
    # Only create one copy to improve performance
    overlay = frame.copy()
    
    # Calculate dimensions once
    width = x2 - x1
    height = y2 - y1
    
    # Fill central areas
    cv2.rectangle(overlay, (x1+radius, y1), (x2-radius, y2), color, cv2.FILLED, cv2.LINE_AA)
    cv2.rectangle(overlay, (x1, y1+radius), (x2, y2-radius), color, cv2.FILLED, cv2.LINE_AA)
    
    # Fill corners
    cv2.circle(overlay, (x1+radius, y1+radius), radius, color, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(overlay, (x2-radius, y1+radius), radius, color, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(overlay, (x1+radius, y2-radius), radius, color, cv2.FILLED, cv2.LINE_AA)
    cv2.circle(overlay, (x2-radius, y2-radius), radius, color, cv2.FILLED, cv2.LINE_AA)
    
    # Blend with original
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
    
    # Centered text if provided
    if text:
        font = FONT_MEDIUM
        
        # Get text dimensions
        text_width, text_height = get_text_size(text, font)
        
        # Calculate center position
        text_x = x1 + (width - text_width) // 2
        text_y = y1 + (height - text_height) // 2 - text_height // 4
        
        # Put text on buttons
        put_text_pil(frame, text, (text_x, text_y), font, (255, 255, 255))

def get_output_filename(base_name):

    # Create output directory if it doesn't exist
    output_dir = 'output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get index based on number of files in output directory
    index = len([f for f in os.listdir(output_dir) if f.startswith(base_name) and f.endswith('.svg')])

    # Construct the full filename
    filename = os.path.join(output_dir, f"{base_name}_{index+1}.svg")
    
    return filename

# Export to SVG
def export_svg(strokes, width, height):
    filename = get_output_filename('drawing')
    fullpath = os.path.abspath(filename)
    dwg = svgwrite.Drawing(filename, size=(width, height))
    for st in strokes:
        pts = st['points']
        if len(pts) > 1:
            dwg.add(dwg.polyline(points=pts,
                                  stroke=st['hex'],
                                  fill='none',
                                  stroke_width=st['thickness']))
    dwg.save()
    return fullpath

# Main application
def main():
    # Initialize fonts and cache before starting the main loop
    initialize_fonts_and_cache()
    
    # Set up video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Set resolution based on architecture
    if IS_ARM_MAC:
        # Lower resolution for better performance on ARM Macs
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)
        print("Optimized settings for ARM Mac applied")
    else:
        # Standard resolution for other platforms
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Set FPS target
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Prepare UI variables
    drawing = False
    strokes = []
    current_color = COLORS[0]
    current_thickness = 8
    slider_active = False
    last_ui_time = 0
    ui_cooldown = 0.3
    
    # Performance tracking
    prev_time = time.time()
    fps_values = []
    frame_count = 0
    avg_fps = 30.0
    
    # Pre-calculate UI elements to avoid recalculation in the loop
    def precompute_ui(width, height):
        ui = {}
        
        # Background ellipse dimensions
        ui['logo_width'] = 200
        ui['logo_x'] = BUTTON_MARGIN
        ui['logo_y'] = BUTTON_MARGIN + BUTTON_HEIGHT // 2
        
        # UI layout starts after program name
        x_start = ui['logo_x'] + ui['logo_width'] + BUTTON_MARGIN * 2
        
        # Color pickers
        cp_positions = []
        x = x_start
        for _ in COLORS:
            cp_positions.append((x, x + COLOR_BOX_SIZE))
            x += COLOR_BOX_SIZE + COLOR_BOX_MARGIN
        ui['cp_positions'] = cp_positions
        
        # Button positions - all in one row to the right of color buttons
        x_buttons = x + BUTTON_MARGIN  # Start right after color buttons
        y_buttons = BUTTON_MARGIN  # Same Y position as colors
        
        # Single row buttons
        ui['undo_x1'], ui['undo_x2'] = x_buttons, x_buttons + BUTTON_WIDTH
        ui['clear_x1'], ui['clear_x2'] = ui['undo_x2'] + BUTTON_MARGIN, ui['undo_x2'] + BUTTON_MARGIN + BUTTON_WIDTH
        ui['save_x1'], ui['save_x2'] = ui['clear_x2'] + BUTTON_MARGIN, ui['clear_x2'] + BUTTON_MARGIN + BUTTON_WIDTH
        ui['exit_x1'], ui['exit_x2'] = ui['save_x2'] + BUTTON_MARGIN, ui['save_x2'] + BUTTON_MARGIN + BUTTON_WIDTH
        
        ui['btn_y1'], ui['btn_y2'] = y_buttons, y_buttons + BUTTON_HEIGHT
        
        # Slider - positioned below all buttons
        ui['sl_x1'] = BUTTON_MARGIN
        ui['sl_x2'] = ui['sl_x1'] + SLIDER_THICKNESS
        ui['sl_y1'] = ui['btn_y2'] + BUTTON_MARGIN
        ui['sl_y2'] = height - BUTTON_MARGIN
        
        # Drawing area - starts right below the top button row
        ui['draw_x1'] = ui['sl_x2'] + BUTTON_MARGIN
        ui['draw_y1'] = ui['sl_y1']
        ui['draw_x2'] = width - BUTTON_MARGIN
        ui['draw_y2'] = height - BUTTON_MARGIN
        
        return ui
    
    # Create window with proper flags for ARM optimizations
    cv2.namedWindow('AirCanvas', cv2.WINDOW_NORMAL)
    
    # MediaPipe optimization settings
    complexity = 0 if IS_ARM_MAC else 1  # Use simplest model on ARM Macs
    hand_confidence = 0.7 if IS_ARM_MAC else 0.8  # Lower confidence for better performance
    
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=hand_confidence,
        min_tracking_confidence=hand_confidence,
        model_complexity=complexity
    ) as hands:
        # Wait for first frame to calculate UI dimensions
        ret, frame = cap.read()
        if not ret:
            print("Could not read initial frame")
            return
            
        # Get the frame dimensions and precompute UI
        h, w, _ = cv2.flip(frame, 1).shape
        ui = precompute_ui(w, h)
        
        # Create and pre-allocate buffers for better performance
        if IS_ARM_MAC:
            rgb_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Performance counter
            frame_count += 1
            
            # Performance measurement (less frequent updates)
            current_time = time.time()
            if current_time - prev_time > 0.5:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time
                fps_values.append(fps)
                if len(fps_values) > 5:
                    fps_values.pop(0)
                avg_fps = sum(fps_values) / max(1, len(fps_values))
            
            # Flip frame horizontally
            frame = cv2.flip(frame, 1)
            
            # Convert BGR to RGB - optimized for ARM
            if IS_ARM_MAC:
                cv2.cvtColor(frame, cv2.COLOR_BGR2RGB, rgb_buffer)
                results = hands.process(rgb_buffer)
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb)
            
            # Background ellipse behind program name
            name_text = 'AirCanvas'
            cv2.ellipse(frame, 
                      (ui['logo_x'] + ui['logo_width']//2, ui['logo_y']), 
                      (ui['logo_width']//2, BUTTON_HEIGHT//2 + 5), 
                      0, 0, 360, (111, 57, 15), cv2.FILLED, cv2.LINE_AA)
            
            # Draw program name using PIL for custom font
            if USE_PIL_FONTS:
                text_width, text_height = get_text_size(name_text, FONT_LARGE)
                frame = draw_text(frame, name_text, 
                                (ui['logo_x'] + (ui['logo_width'] - text_width)//2, 
                                 ui['logo_y'] - int(3*text_height//4)), 
                                FONT_LARGE, (255, 255, 255))
            else:
                cv2.putText(frame, name_text, (ui['logo_x'] + 20, ui['logo_y'] + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
            
            # Draw UI elements
            for i, col in enumerate(COLORS):
                x1, x2 = ui['cp_positions'][i]
                draw_button(frame, x1, ui['btn_y1'], x2, ui['btn_y2'], col['bgr'], '', alpha=0.8)
                if col == current_color:
                    cv2.rectangle(frame, (x1, ui['btn_y1']), (x2, ui['btn_y2']), (255,255,255), 3, cv2.LINE_AA)

            # Draw buttons with Undo button added
            draw_button(frame, ui['undo_x1'], ui['btn_y1'], ui['undo_x2'], ui['btn_y2'], (80,80,80), 'Undo', alpha=0.8)
            draw_button(frame, ui['clear_x1'], ui['btn_y1'], ui['clear_x2'], ui['btn_y2'], (80,80,80), 'Clear', alpha=0.8)
            draw_button(frame, ui['save_x1'], ui['btn_y1'], ui['save_x2'], ui['btn_y2'], (80,80,80), 'Save', alpha=0.8)
            draw_button(frame, ui['exit_x1'], ui['btn_y1'], ui['exit_x2'], ui['btn_y2'], (80,80,80), 'Exit', alpha=0.8)

            # Draw slider track with rounded corners
            draw_button(frame, ui['sl_x1'], ui['sl_y1'], ui['sl_x2'], ui['sl_y2'], (80,80,80), '', alpha=0.8, radius=SLIDER_THICKNESS//2)
            
            # Draw slider knob
            ratio = (current_thickness - THICKNESS_MIN) / (THICKNESS_MAX - THICKNESS_MIN)
            ky = int(ui['sl_y2'] - ratio * (ui['sl_y2'] - ui['sl_y1']))
            cv2.circle(frame, ((ui['sl_x1']+ui['sl_x2'])//2, ky), SLIDER_THICKNESS//2, (255,255,255), cv2.FILLED, cv2.LINE_AA)

            # Draw drawing area
            cv2.rectangle(frame, (ui['draw_x1'], ui['draw_y1']), (ui['draw_x2'], ui['draw_y2']), (176,98,37), 2, cv2.LINE_AA)

            # Add "DRAWING AREA" text
            if USE_PIL_FONTS:
                text_width, text_height = get_text_size('DRAWING AREA', FONT_MEDIUM)
                frame = draw_text(frame, 'DRAWING AREA', 
                                (ui['draw_x1'] + 10, ui['draw_y1'] + 10), 
                                FONT_MEDIUM, (176,98,37))
            else:
                cv2.putText(frame, 'DRAW HERE', (ui['draw_x1']+10, ui['draw_y1']+30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2, cv2.LINE_AA)
            
            # Show FPS
            fps_text = f"FPS: {avg_fps:.1f}"
            if IS_ARM_MAC:
                fps_text += " (ARM)"
            (fps_w, fps_h), _ = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.putText(frame, fps_text, (w - fps_w - 20, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            # Process hand landmarks
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                
                # Custom function to draw landmarks with different colors
                def draw_custom_landmarks(image, landmark_list):
                    # Define colors for each landmark (BGR format)
                    # You can change these colors to whatever you want
                    landmark_colors = [
                        (80, 80, 80),    # WRIST (0)
                        (80, 80, 80),   # THUMB_CMC (1)
                        (80, 80, 80),  # THUMB_MCP (2)
                        (242, 255, 0),  # THUMB_IP (3)
                        (255, 255, 255),  # THUMB_TIP (4)
                        (80, 80, 80),  # INDEX_MCP (5)
                        (242, 255, 0),  # INDEX_PIP (6)
                        (242, 255, 0),  # INDEX_DIP (7)
                        (0, 0, 255),  # INDEX_TIP (8)
                        (80, 80, 80),  # MIDDLE_MCP (9)
                        (80, 80, 80),  # MIDDLE_PIP (10)
                        (80, 80, 80),  # MIDDLE_DIP (11)
                        (80, 80, 80),  # MIDDLE_TIP (12)
                        (80, 80, 80),  # RING_MCP (13)
                        (80, 80, 80),  # RING_PIP (14)
                        (80, 80, 80),  # RING_DIP (15)
                        (80, 80, 80),  # RING_TIP (16)
                        (80, 80, 80),  # PINKY_MCP (17)
                        (80, 80, 80),  # PINKY_PIP (18)
                        (80, 80, 80),  # PINKY_DIP (19)
                        (80, 80, 80),  # PINKY_TIP (20)
                    ]
                    
                    # Draw landmarks
                    h, w, _ = image.shape
                    landmarks = landmark_list.landmark
                    
                    # Draw each landmark as a colored circle
                    for idx, landmark in enumerate(landmarks):
                        # Convert normalized coordinates to pixel coordinates
                        x, y = int(landmark.x * w), int(landmark.y * h)
                        
                        # Draw the landmark
                        if idx == 4 or idx == 8:
                            cv2.circle(image, (x, y), 5, landmark_colors[idx], cv2.FILLED)
                        else:
                            cv2.circle(image, (x, y), 3, landmark_colors[idx], cv2.FILLED)
                        
                        # Optional: Add landmark index for debugging
                        # cv2.putText(image, str(idx), (x+10, y+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, landmark_colors[idx], 2)
                    
                    return image
                
                # Custom function to draw connections with different colors
                def draw_custom_connections(image, landmark_list):
                    # Define colors for different parts of the hand (BGR format)
                    connection_colors = {
                        'thumb': (242, 255, 0),     # Orange
                        'index': (242, 255, 0),       # Green
                        'middle': (80,80,80),    # Yellow
                        'ring': (80,80,80),      # Magenta
                        'pinky': (80,80,80),     # Purple
                        'palm': (80,80,80)     # White
                    }
                    
                    # Define connections and their finger groups
                    connections = [
                        # Thumb connections
                        ([0, 1], 'palm'), ([1, 2], 'palm'), ([2, 3], 'thumb'), ([3, 4], 'thumb'),
                        # Index finger connections
                        ([0, 5], 'palm'), ([5, 6], 'index'), ([6, 7], 'index'), ([7, 8], 'index'),
                        # Middle finger connections
                        ([0, 9], 'palm'), ([9, 10], 'middle'), ([10, 11], 'middle'), ([11, 12], 'middle'),
                        # Ring finger connections
                        ([0, 13], 'palm'), ([13, 14], 'ring'), ([14, 15], 'ring'), ([15, 16], 'ring'),
                        # Pinky finger connections
                        ([0, 17], 'palm'), ([17, 18], 'pinky'), ([18, 19], 'pinky'), ([19, 20], 'pinky'),
                        # Palm connections
                        ([5, 9], 'palm'), ([9, 13], 'palm'), ([13, 17], 'palm')
                    ]
                    
                    # Draw connections
                    h, w, _ = image.shape
                    landmarks = landmark_list.landmark
                    
                    # For each connection
                    for connection, finger_type in connections:
                        # Get connection points
                        start_idx, end_idx = connection
                        start_point = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
                        end_point = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
                        
                        # Draw the connection
                        color = connection_colors[finger_type]
                        cv2.line(image, start_point, end_point, color, 1, cv2.LINE_AA)
                    
                    return image
                
                # Draw the hand with custom colors
                draw_custom_connections(frame, hand)
                draw_custom_landmarks(frame, hand)
                
                # Extract index finger tip position for UI interaction
                ui_lm = hand.landmark[INDEX_TIP]
                ui_x, ui_y = int(ui_lm.x * w), int(ui_lm.y * h)
                now = time.time()

                # UI interaction with cooldown
                if now - last_ui_time > ui_cooldown:
                    # Color selectors
                    for i, col in enumerate(COLORS):
                        x1, x2 = ui['cp_positions'][i]
                        if x1 <= ui_x <= x2 and ui['btn_y1'] <= ui_y <= ui['btn_y2']:
                            current_color = col
                            last_ui_time = now
                    
                    # Button interactions
                    if ui['undo_x1'] <= ui_x <= ui['undo_x2'] and ui['btn_y1'] <= ui_y <= ui['btn_y2']:
                        if strokes:
                            strokes.pop()  # Remove the last stroke
                        last_ui_time = now
                    
                    if ui['clear_x1'] <= ui_x <= ui['clear_x2'] and ui['btn_y1'] <= ui_y <= ui['btn_y2']:
                        strokes.clear()
                        last_ui_time = now
                    
                    if ui['save_x1'] <= ui_x <= ui['save_x2'] and ui['btn_y1'] <= ui_y <= ui['btn_y2']:
                        path = export_svg(strokes, w, h)
                        print(f"Saved: {path}")
                        last_ui_time = now
                    
                    if ui['exit_x1'] <= ui_x <= ui['exit_x2'] and ui['btn_y1'] <= ui_y <= ui['btn_y2']:
                        print('Exit pressed.')
                        break
                    
                    # Slider activation
                    if ui['sl_x1'] <= ui_x <= ui['sl_x2'] and ui['sl_y1'] <= ui_y <= ui['sl_y2']:
                        slider_active = True

                # Slider handling
                if slider_active:
                    if ui['sl_x1'] <= ui_x <= ui['sl_x2'] and ui['sl_y1'] <= ui_y <= ui['sl_y2']:
                        ratio = (ui['sl_y2'] - ui_y) / (ui['sl_y2'] - ui['sl_y1'])
                        current_thickness = int(THICKNESS_MIN + max(0, min(1, ratio)) * (THICKNESS_MAX - THICKNESS_MIN))
                    else:
                        slider_active = False

                # Drawing gesture detection
                mid_x = mid_y = None
                if is_drawing_gesture(hand):
                    thumb = hand.landmark[THUMB_TIP]
                    idx = hand.landmark[INDEX_TIP]
                    mid_x = int((thumb.x + idx.x)/2 * w)
                    mid_y = int((thumb.y + idx.y)/2 * h)

                # Drawing handling
                if mid_x and mid_y and mid_x > ui['draw_x1'] and mid_y > ui['draw_y1'] and is_drawing_gesture(hand):
                    if not drawing:
                        drawing = True
                        current_stroke = {'points': [], 'hex': current_color['hex'], 'thickness': current_thickness, 'color':current_color['bgr']}
                        strokes.append(current_stroke)
                    current_stroke['points'].append((mid_x, mid_y))
                    clear_start = None
                else:
                    if drawing:
                        drawing = False

            # Render strokes - use numpy operations for ARM Macs when possible
            for st in strokes:
                points = st['points']
                if len(points) > 1:
                    if IS_ARM_MAC and len(points) > 100:
                        # For long strokes, use batch processing (more efficient on ARM)
                        points_array = np.array(points)
                        for i in range(1, len(points_array)):
                            cv2.line(frame, 
                                   (points_array[i-1][0], points_array[i-1][1]), 
                                   (points_array[i][0], points_array[i][1]), 
                                   st['color'], st['thickness'])
                    else:
                        # For shorter strokes, regular processing is fine
                        for i in range(1, len(points)):
                            cv2.line(frame, points[i-1], points[i], st['color'], st['thickness'])
            
            cv2.imshow('AirCanvas', frame)
            
            # Check for keyboard shortcuts
            key = cv2.waitKey(1) & 0xFF
            
            # Ctrl+Z for undo
            if key == 26:  # Ctrl+Z (ASCII value 26)
                if strokes:
                    strokes.pop()
            
            # Exit on ESC key
            if key == 27:  # ESC key
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()