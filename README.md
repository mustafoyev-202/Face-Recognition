# 👤 Real-time Face Recognition

A real-time face recognition system using DeepFace and OpenCV. This application captures video from your webcam and continuously compares detected faces with a reference image, providing instant match feedback.

## ✨ Features

- Real-time face recognition using webcam feed
- Support for multiple video capture backends
- Thread-safe face verification
- Performance optimization through frame sampling
- Visual feedback with color-coded matching status
- Multi-platform support (Windows, Linux, MacOS)

## 🛠️ Technologies Used

- Python 3.x
- OpenCV (cv2)
- DeepFace
- Threading

## 📋 Prerequisites

- Python 3.7 or higher
- Webcam
- Reference image for face comparison

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/mustafoyev-202/Face-Recognition.git
cd face-recognition
```

2. Install required packages:
```bash
pip install opencv-python
pip install deepface
```

3. Place your reference image:
- Add an image named `Image.jpg` in the project root directory
- This will be the face that the system tries to match against

## 💻 Usage

1. Run the program:
```bash
python main.py
```

2. The application will:
- Open your webcam
- Start comparing captured frames with the reference image
- Display "MATCH" (green) or "NO MATCH" (red) on the video feed

3. To exit:
- Press 'q' to close the application

## 🎯 How It Works

1. **Camera Initialization**
   - Attempts to open the camera using different backends (DirectShow, Media Foundation, V4L2)
   - Sets resolution to 640x480 for optimal performance

2. **Face Recognition Process**
   - Captures frames from the webcam
   - Processes every 30th frame to maintain performance
   - Uses threading for non-blocking face verification
   - Compares captured faces with the reference image using DeepFace

3. **Visual Feedback**
   - Displays real-time matching status on the video feed
   - Green "MATCH" text indicates a successful match
   - Red "NO MATCH" text indicates no match

## ⚙️ Configuration

You can modify these parameters in `main.py`:

```python
# Frame processing interval
if counter % 30 == 0:  # Adjust for different processing frequencies

# Video resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Change width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Change height
```

## 🔍 Troubleshooting

1. **Camera Access Issues**
   - The program attempts multiple backends (DirectShow, Media Foundation, V4L2)
   - Ensure your webcam is not being used by another application
   - Check if your webcam is properly connected

2. **Reference Image Problems**
   - Ensure `Image.jpg` exists in the project directory
   - Verify the image format is supported
   - Check if the image contains a clear, well-lit face

3. **Performance Issues**
   - Adjust the frame processing interval (currently every 30th frame)
   - Lower the resolution if needed
   - Ensure adequate system resources are available

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is distributed under the MIT License. See `LICENSE` file for more information.

## ⚠️ Important Notes

- The system's accuracy depends on lighting conditions and camera quality
- Face recognition is performed using DeepFace's verify function
- The application uses threading to prevent UI freezing during face verification

## 📮 Contact

Name - baxtiyormustafoyev2006@gmail.com
Project Link: https://github.com/mustafoyev-202/Face-Recognition