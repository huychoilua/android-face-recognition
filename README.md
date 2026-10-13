# Android Face Recognition

Academic Android face-recognition project by Nguyễn Đại Minh Huy.

## Features and technologies
- Android Java UI with CameraX front-camera capture.
- OpenCV YuNet face detection and SFace feature extraction using pretrained ONNX models.
- Face alignment and 128-dimensional embedding comparison with cosine similarity.
- ML Kit eye-closure check before recognition (a basic demonstration, not a complete anti-spoofing system).
- Registration/login and recognition history stored locally with Room/SQLite.
- Background executors for camera analysis and database operations.
- Python/OpenCV/NumPy script for generating reference embeddings.

## Pipeline
Reference image → YuNet detection → face alignment → SFace embedding → local reference file.

Camera frame → eye-closure check → face detection/alignment → embedding → similarity comparison → recognition history.

## Repository status
This repository contains the available `app/src/main` source and Python script. The original archive did not include Gradle build files, the wrapper, or dependency versions. It is a source-code portfolio snapshot and is not yet a complete buildable Android Studio project. No build or runtime validation is claimed.

## Models and sample data
Model binaries, personal images, and biometric embedding files are intentionally excluded. Obtain YuNet and SFace from the [official OpenCV Zoo](https://github.com/opencv/opencv_zoo), observing their respective model licenses. Expected Android asset filenames:
- `face_detection_yunet.onnx`
- `face_recognition_sface_2021dec.onnx`

For an authorized local demonstration, generate your own reference vectors and update the demo entries in `MainActivity.java`. The example expects `demo_001_embedding.bin`, `demo_002_embedding.bin`, and `demo_003_embedding.bin`. Run the Python script only after installing compatible OpenCV and NumPy packages and updating its image/model paths.

## Limitations
- Models are pretrained; this project integrates inference and does not claim model training.
- Login is an academic prototype; passwords are currently stored without hashing and must be hardened before production use.
- Eye-closure detection is a simplified check.
- The Python script selects the first detected face; use single-person reference images.

Third-party libraries and models retain their original licenses.
