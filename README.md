# 1. Introduction
## ObjectDetection

Sliding Windows for Object Detection with Python and OpenCV

This is a work taken from " <https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/> "
for study purposes.
The python command to run this project is python sliding_window.py --image images/adrian_florida.jpg

Pyramids and sliding window are used for detecting objects from an image. Here we are implementing a sliding window for images with the help of pyramid function. Sliding window is a window that has a fixed width and height which slides over the image. Hence we can use a image recognition algorithm along with this to find whether the window contains an image. Image pyramid helps us to find the multi scale representation of images. The images can be splitted out into a number of layers , the base layer contains our full image and when it goes up we get a smaller and smoothened image.
These sliding windows together with image pyramid helps us to find the objects in images

Reference: <https://www.pyimagesearch.com>

# 2. Prerequisites

- make ```sudo apt install make```
- python3-dev python3-pip ```sudo apt install python3-dev python3-pip```
- Python >= 3.8

# 3. Document

# 4. Installation

- Install the requirements inside of a Python virtualenv (recommend)
```BASH
    pip install virtualenv
    virtualenv -p python3.8 venv
    source venv/bin/activate
```

- Make requirements
```BASH
    make requirements
```

- Run
```
    python sliding_window.py --image images/stick_of_truth.jpg 
```

# 5. Example rasa commands

# 6. Debug
