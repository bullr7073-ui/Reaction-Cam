# 🎭 Reaction-Cam

A real-time face tracking project built with **Python and MediaPipe** that detects facial expressions through your webcam and responds with different reaction images.

The project uses face landmarks and facial expression analysis to turn your camera feed into an interactive reaction system.

---

## ✨ Features

- 🎥 Real-time webcam face tracking
- 👁️ Facial landmark detection
- 😄 Expression-based reactions
- 🖼️ Displays different reaction images based on detected expressions
- 🎯 Calibration system for improving detection
- ⚡ Real-time processing
- 🐍 Built with Python
- 🧠 Powered by MediaPipe Face Landmarker

---

## 🖼️ Reaction Examples

The project currently includes reaction images such as:

| Reaction | Image |
|----------|-------|
| 🥰 Cute | `cute.jpeg` |
| 🐶 Dog | `dog.jpeg` |
| 🤪 Goofy | `goofy.jpeg` |
| 🐈 Larry | `larry.jpeg` |
| 🐒 Monkey | `monkey.jpeg` |
| 😏 Rizz | `rizz.jpeg` |
| 😁 Ronaldo | `ronaldo.jpeg` |
| 😮 What | `what.jpeg` |

---

## 🛠️ Tech Stack

- **Python**
- **MediaPipe**
- **OpenCV**
- **NumPy**
- **Webcam / OpenCV Video Capture**

---

## 📁 Project Structure

```text
Reaction-Cam/
│
├── calibration/
│   └── calibration.py
│
├── reactions/
│   ├── cute.jpeg
│   ├── dog.jpeg
│   ├── goofy.jpeg
│   ├── larry.jpeg
│   ├── monkey.jpeg
│   ├── rizz.jpeg
│   ├── ronaldo.jpeg
│   └── what.jpeg
│
├── face_landmarker.task
├── main.py
├── .gitignore
└── README.md