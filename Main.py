import cv2
import mediapipe as mp
import time
import svgwrite
import os

# Suppress TensorFlow/MediaPipe logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Gesture thresholds
PINCH_THRESHOLD = 0.05
ALL_FINGERS_THRESH_TIME = 3.0
THUMBS_UP_HOLD_TIME = 2.0

# UI constants
BUTTON_WIDTH = 150
BUTTON_HEIGHT = 80
BUTTON_MARGIN = 20
COLOR_BOX_SIZE = 80
COLOR_BOX_MARGIN = 20
SLIDER_THICKNESS = 30

# Define available colors
COLORS = [
    {'name': 'Black', 'bgr': (0, 0, 0), 'hex': '#000000'},
    {'name': 'Red',   'bgr': (0, 0, 255), 'hex': '#FF0000'},
    {'name': 'Green', 'bgr': (0, 255, 0), 'hex': '#00FF00'},
    {'name': 'Blue',  'bgr': (255, 0, 0), 'hex': '#0000FF'}
]

# Thickness range
THICKNESS_MIN = 1
THICKNESS_MAX = 10

# Landmark indices
INDEX_TIP = 8
THUMB_TIP = 4
MIDDLE_TIP, RING_TIP, PINKY_TIP = 12, 16, 20
FINGER_TIPS = [INDEX_TIP, MIDDLE_TIP, RING_TIP, PINKY_TIP]

# Normalize distance
def norm_dist(a, b):
    return ((a.x - b.x)**2 + (a.y - b.y)**2)**0.5

# Gesture detectors
def is_drawing_gesture(hand):
    thumb, index = hand.landmark[THUMB_TIP], hand.landmark[INDEX_TIP]
    return norm_dist(thumb, index) < PINCH_THRESHOLD

def all_fingers_stretched(hand):
    for tip_id in FINGER_TIPS:
        tip = hand.landmark[tip_id]
        pip = hand.landmark[tip_id - 2]
        if tip.y > pip.y:
            return False
    return True

def is_thumbs_up(hand):
    thumb_tip = hand.landmark[THUMB_TIP]
    thumb_ip = hand.landmark[THUMB_TIP - 1]
    if thumb_tip.y > thumb_ip.y:
        return False
    for tip_id in FINGER_TIPS:
        tip = hand.landmark[tip_id]
        mcp = hand.landmark[tip_id - 3]
        if tip.y < mcp.y:
            return False
    return True

# Export to SVG
def export_svg(strokes, w, h):
    filename = 'signature.svg'
    fullpath = os.path.abspath(filename)
    dwg = svgwrite.Drawing(filename, size=(w, h))
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
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print('Error: Could not open webcam.')
        return

    drawing = False
    strokes = []
    current_stroke = None
    current_color = COLORS[0]
    current_thickness = 2
    clear_start = None
    thumbs_start = None
    slider_active = False
    last_ui_time = 0
    ui_cooldown = 0.3

    cv2.namedWindow('Signature Drawer', cv2.WINDOW_NORMAL)

    with mp_hands.Hands(max_num_hands=1,
                        min_detection_confidence=0.7,
                        min_tracking_confidence=0.7) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            # Define top UI row
            # Color pickers left to right
            cp_positions = []
            x = BUTTON_MARGIN
            for _ in COLORS:
                cp_positions.append((x, x + COLOR_BOX_SIZE))
                x += COLOR_BOX_SIZE + COLOR_BOX_MARGIN
            # Clear, Save, Exit buttons
            clear_x1 = x; clear_x2 = clear_x1 + BUTTON_WIDTH
            save_x1 = clear_x2 + BUTTON_MARGIN; save_x2 = save_x1 + BUTTON_WIDTH
            exit_x1 = save_x2 + BUTTON_MARGIN; exit_x2 = exit_x1 + BUTTON_WIDTH
            btn_y1 = BUTTON_MARGIN; btn_y2 = btn_y1 + BUTTON_HEIGHT

            # Vertical slider on left border with extra margin
            sl_x1 = BUTTON_MARGIN * 2  # margin from left border
            sl_x2 = sl_x1 + SLIDER_THICKNESS
            sl_y1 = btn_y2 + BUTTON_MARGIN
            sl_y2 = h - BUTTON_MARGIN

            # Drawing zone: padded from slider and top UI
            draw_x1 = sl_x2 + BUTTON_MARGIN  # left padding from slider
            draw_y1 = sl_y1  # top padding below slider
            draw_x2 = w - BUTTON_MARGIN
            draw_y2 = h

            # Draw UI
            # Color pickers
            for i, col in enumerate(COLORS):
                x1, x2 = cp_positions[i]
                cv2.rectangle(frame, (x1, btn_y1), (x2, btn_y2), col['bgr'], cv2.FILLED)
                if col == current_color:
                    cv2.rectangle(frame, (x1, btn_y1), (x2, btn_y2), (255,255,255), 5)
            # Clear button
            cv2.rectangle(frame, (clear_x1, btn_y1), (clear_x2, btn_y2), (80,80,80), cv2.FILLED)
            cv2.putText(frame, 'Clear', (clear_x1+20, btn_y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            # Save button
            cv2.rectangle(frame, (save_x1, btn_y1), (save_x2, btn_y2), (80,80,80), cv2.FILLED)
            cv2.putText(frame, 'Save', (save_x1+30, btn_y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255,255,255), 2)
            # Exit button
            cv2.rectangle(frame, (exit_x1, btn_y1), (exit_x2, btn_y2), (50,50,50), cv2.FILLED)
            cv2.putText(frame, 'Exit', (exit_x1+20, btn_y1+50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 3)

            # Slider track
            cv2.rectangle(frame, (sl_x1, sl_y1), (sl_x2, sl_y2), (200,200,200), cv2.FILLED)
            # Slider knob
            ratio = (current_thickness - THICKNESS_MIN) / (THICKNESS_MAX - THICKNESS_MIN)
            ky = int(sl_y2 - ratio * (sl_y2 - sl_y1))
            cv2.circle(frame, ( (sl_x1+sl_x2)//2, ky ), SLIDER_THICKNESS//2, (100,100,100), cv2.FILLED)
            cv2.putText(frame, f'{current_thickness}', (sl_x2+10, ky+5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

            # Draw 'DRAW HERE' box
            cv2.rectangle(frame, (draw_x1, draw_y1), (draw_x2, draw_y2), (0,255,0), 2)
            cv2.putText(frame, 'DRAW HERE', (draw_x1+10, draw_y1+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            # Process hand
            if results.multi_hand_landmarks:
                hand = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
                x = int(hand.landmark[INDEX_TIP].x * w)
                y = int(hand.landmark[INDEX_TIP].y * h)
                now = time.time()

                # UI taps by index
                if now - last_ui_time > ui_cooldown:
                    # Colors
                    for i, col in enumerate(COLORS):
                        x1, x2 = cp_positions[i]
                        if x1 <= x <= x2 and btn_y1 <= y <= btn_y2:
                            current_color = col; last_ui_time = now
                    # Clear
                    if clear_x1 <= x <= clear_x2 and btn_y1 <= y <= btn_y2:
                        strokes.clear(); last_ui_time = now
                    # Save
                    if save_x1 <= x <= save_x2 and btn_y1 <= y <= btn_y2:
                        path = export_svg(strokes, w, h); print(f'Saved: {path}'); last_ui_time = now
                    # Exit
                    if exit_x1 <= x <= exit_x2 and btn_y1 <= y <= btn_y2:
                        print('Exit pressed.'); break
                    # Slider activate
                    if sl_x1 <= x <= sl_x2 and sl_y1 <= y <= sl_y2:
                        slider_active = True
                # Slider drag: only while index stays within slider bounds
                if slider_active:
                    if sl_x1 <= x <= sl_x2 and sl_y1 <= y <= sl_y2:
                        # adjust thickness based on vertical position
                        ratio = (sl_y2 - y) / (sl_y2 - sl_y1)
                        ratio = max(0, min(1, ratio))
                        current_thickness = int(THICKNESS_MIN + ratio * (THICKNESS_MAX - THICKNESS_MIN))
                    else:
                        slider_active = False

                # Drawing only in draw area
                if slider_active:
                    if sl_y1 <= y <= sl_y2:
                        ratio = (sl_y2 - y) / (sl_y2 - sl_y1)
                        ratio = max(0, min(1, ratio))
                        current_thickness = int(THICKNESS_MIN + ratio * (THICKNESS_MAX - THICKNESS_MIN))
                    else:
                        slider_active = False

                # Drawing only in draw area
                if is_drawing_gesture(hand) and y > draw_y1:
                    if not drawing:
                        drawing = True
                        current_stroke = {'points': [], 'hex': current_color['hex'],
                                          'thickness': current_thickness, 'color': current_color['bgr']}
                        strokes.append(current_stroke)
                    current_stroke['points'].append((x, y)); clear_start = None
                else:
                    if drawing:
                        drawing = False
                    # Clear gesture
                    if all_fingers_stretched(hand):
                        if clear_start is None:
                            clear_start = time.time()
                        elif now - clear_start > ALL_FINGERS_THRESH_TIME:
                            strokes.clear(); clear_start = None
                    else:
                        clear_start = None
                    # Save gesture
                    if is_thumbs_up(hand):
                        if thumbs_start is None:
                            thumbs_start = now
                        elif now - thumbs_start > THUMBS_UP_HOLD_TIME:
                            path = export_svg(strokes, w, h)
                            print(f'Saved signature: {path}')
                            thumbs_start = None
                    else:
                        thumbs_start = None

            # Render strokes
            for st in strokes:
                for i in range(1, len(st['points'])):
                    cv2.line(frame, st['points'][i-1], st['points'][i], st['color'], st['thickness'])

            cv2.imshow('Signature Drawer', frame)
            if cv2.waitKey(1) & 0xFF == 27:
                print('Exiting.'); break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
