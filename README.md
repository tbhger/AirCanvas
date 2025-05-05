# 🎨 AirCanvas - Draw in the Air with Your Fingers

<div align="center">
  <img src="preview.gif" alt="AirCanvas Demo" width="800"/>
  <p><em>Transform your fingers into digital paintbrushes with AirCanvas</em></p>
</div>

## ✨ Overview

AirCanvas is a drawing application that turns your webcam and hands into a virtual drawing canvas. Using MediaPipe's advanced hand-tracking technology, you can create digital art using intuitive finger gestures - no stylus or mouse required!

### 🚀 Key Features

- **Finger Drawing**: Use thumb-index pinch gesture to draw in the air
- **Color Palette**: Choose from 4 pre-configured colors
- **Adjustable Brush**: Dynamic thickness control with vertical slider
- **SVG Export**: Save your creations as vector graphics
- **Performance Optimized**: Specially tuned for ARM-based MacBooks
- **Elegant UI**: Custom font support and polished visual elements

## 🛠️ Technology Stack

- **Python 3.x**
- **OpenCV**: Real-time video processing
- **MediaPipe**: Advanced hand landmark detection
- **PIL (Pillow)**: Custom font rendering
- **svgwrite**: Vector graphics export
- **NumPy**: Optimized mathematical operations

## 📋 Prerequisites

- Python 3.11.x or higher
- A working webcam
- Git (for cloning the repository)

## 🔧 Installation

1. Clone the repository:
```bash
git clone https://github.com/tbh_ger/AirCanvas.git
cd AirCanvas
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Add a custom font (optional):
   - Place your preferred .ttf font file in the `fonts/` directory
   - Update the `FONT_PATH` variable in the code

5. Run the application:
```bash
python aircanvas.py
```

## 🎯 How to Use

### Basic Drawing
1. Position your hand in front of the webcam
2. Create the "pinch" gesture (thumb tip touching index finger tip)
3. Move your pinched fingers to draw
4. Release the pinch to stop drawing

### UI Controls
- **Color Selection**: Tap on color squares at the top
- **Brush Size**: Slide the vertical bar on the left
- **Clear Canvas**: Tap the 'Clear' button
- **Save Drawing**: Tap the 'Save' button
- **Exit**: Tap the 'Exit' button or press ESC

### Gesture Controls
- **Drawing Gesture**: Pinch (thumb + index finger)
- **UI Interaction**: Point with index finger

## 💡 Technical Highlights

### Performance Optimization
- ARM MacBook-specific optimizations for improved performance
- Vectorized operations using NumPy
- Pre-allocated buffers for image processing
- Text caching for UI elements
- ROI-based text rendering

### Hand Detection
- Custom landmark visualization with color-coded joints
- Optimized tracking for single-hand operation
- Reduced detection confidence for better performance

### File Export
- Automatic incrementing filenames
- SVG format for scalability
- Preserves stroke order and properties

## 📁 Project Structure

```
aircanvas/
├── aircanvas.py          # Main application
├── requirements.txt      # Python dependencies
├── fonts/               # Custom fonts directory
│   └── Bauhaus93.ttf    # Example custom font
├── output/              # SVG export location
│   └── drawing_*.svg    # Generated drawings
└── README.md            # This file
```

## 🎨 Customization

### Color Palette
Modify the `COLORS` list to add new colors:
```python
COLORS = [
    {'name': 'Purple', 'bgr': (128, 0, 128), 'hex': '#800080'},
    # Add more colors here
]
```

### Brush Settings
Adjust thickness range in constants:
```python
THICKNESS_MIN = 1
THICKNESS_MAX = 15
```

### UI Layout
Modify button sizes and positions:
```python
BUTTON_WIDTH = 200
BUTTON_HEIGHT = 70
```

## 🐛 Troubleshooting

### Common Issues

1. **Camera not detected**
   - Check camera permissions
   - Try different camera indices (0, 1, 2...)

2. **Low frame rate**
   - Adjust resolution settings
   - Check ARM Mac optimizations are enabled

3. **Font not loading**
   - Place font file in `fonts/` directory
   - Verify font path is correct

4. **Hand detection issues**
   - Ensure good lighting
   - Keep hand within frame
   - Adjust confidence thresholds

## 📧 Contact

Created by [Tim Benjamin Hoffmann] - [@tbhger](https://github.com/tbhger)

Project Link: [https://github.com/tbhger/AirCanvas](https://github.com/tbhger/aircanvas)

---

<div align="center">
  <strong>Made with ❤️ and ✋</strong>
</div>