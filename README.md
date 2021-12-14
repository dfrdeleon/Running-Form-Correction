# Running-Form-Correction
Utilizes Pose Estimation to offer sprinters cues based on an image of their running form.

## How to Run

### Dependencies
You will need the dependencies listed below:
Note: it is encouraged that you utilize a venv through either pip or anaconda
- python3
- tensorflow 1.3
- opencv3
- protobuf
- python3-tk

### Install
```bash
$ git clone https://github.com/dfrdeleon/Running-Form-Correction
$ cd Running-Form-Correction
$ pip3 install -r requirements.txt
```

### Pose Estimation Demo

To see an example of the pose estimation overlayed on top of the original image, run the code below. 
Note: Inference generation works best if only one person and their entire body is in frame.

You can set model equal to one of the networks listed below:
- cmu
- dsconv
- mobilenet
  - mobilenet_fast
  - mobilenet_accurate

Make sure to set the imgpath to that of the input frame on your machine.

### Form Correction

Similar to the Pose Estimation Demo, set the intended model and image path:

```
$ python3 form_correction.py --model=cmu --imgpath=...
```
