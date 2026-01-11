
# CNN Hand Gesture Detection using ResNet18

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/pytorch-1.9%2B-EE4C2C)

---

## Overview

This project implements a real-time hand gesture detection system using a Convolutional Neural Network (CNN) based on ResNet18. The model classifies hand gestures for Rock, Paper, and Scissors using webcam input. It leverages PyTorch for deep learning, OpenCV for image capture, and Jupyter Notebook for experimentation.

## Features
- Real-time hand gesture detection via webcam
- Uses ResNet18 architecture, fine-tuned for 3 classes
- Pre-trained model weights included
- Easy-to-use Python script
- Jupyter Notebook for training and experimentation


## Installation

1. **Clone the repository**
	```bash
	git clone https://github.com/DPHeshanRanasinghe/CNN-Hand-Gesture-Detection-using-ResNet18.git
	cd CNN-Hand-Gesture-Detection-using-ResNet18
	```
2. **Install dependencies**
	```bash
	pip install torch torchvision opencv-python pillow numpy
	```
3. **(Optional) Jupyter Notebook**
	```bash
	pip install notebook
	```

## Usage

### Run the Detector
```bash
python main.py
```
Press 'q' to quit the webcam window.

### Train or Experiment
Open `model.ipynb` in Jupyter Notebook for training or further experimentation.

## File Structure

- `main.py` : Main script for real-time detection
- `model.ipynb` : Notebook for model training and analysis
- `resnet18_rps_best_model.pth` : Pre-trained model weights
- `README.md` : Project documentation

## Model Details

- **Architecture:** ResNet18 (modified for 3 output classes)
- **Classes:** Rock, Paper, Scissors
- **Input:** 224x224 RGB images (Region of Interest from webcam)
- **Frameworks:** PyTorch, Torchvision

## How it Works
1. Loads the trained ResNet18 model
2. Captures video from webcam
3. Extracts a region of interest (ROI) for hand detection
4. Preprocesses the image (resize, normalize)
5. Runs inference and displays the predicted gesture and confidence

## Example Output

```
rock (98.2%)
paper (95.7%)
scissors (92.4%)
```

## Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

## Acknowledgements
- [PyTorch](https://pytorch.org/)
- [OpenCV](https://opencv.org/)
- [Torchvision](https://pytorch.org/vision/stable/index.html)
- [Shields.io](https://shields.io/)

## Author

**Heshan Ranasinghe**  
Electronic and Telecommunication Engineering Undergraduate

- Email: hranasinghe505@gmail.com
- GitHub: [@DPHeshanRanasinghe](https://github.com/DPHeshanRanasinghe)
- LinkedIn: [Heshan Ranasinghe](https://www.linkedin.com/in/heshan-ranasinghe-988b00290)

---

