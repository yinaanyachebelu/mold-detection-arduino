# mold-detection-arduino
Detecting Household Mold  Using Transfer Learning, Edge Impulse, TensorFlow Lite and Arduino

### Overview

This project aims to use deep learning on an edge device to detect the growth of mold in homes.  For this project, I employ transfer learning drawing from the MobileNet architecture to deploy a model using an Arduino Nano 33 BLE Sense with an OV7675 camera. In the final deployment of the device, when an image is captured (every 4 seconds), a blue light switches on and off.  If the final decision is that there is no mold, then the green light is turned on; if not, a red light turns on. This is an early-stage prototype for a detection device that would be helpful for homeowners and tenants to monitor the success of a mold remediation service.

To test out the device, you will need the an Arduino Nano 33 BLE Sense with an OV7675 camera, Arduino IDE with the Arduino_OV767X (version 0.02) library installed as well as the inferencing library (found in this repository) which must be added as a .zip library. Use the deployment.ino file located in the **deployment folder** in the **nano_ble33_sense folder** of the inference-library to test out the device.

### Resources that can be found in this repository:

Dataset Folder: Includes the training dataset created from photos manually taken with the Arduino device and split into two folders (mold and no_mold)

model_training.py : Python script used to train the model. Note that this script was developed by [Edge Impulse](https://www.edgeimpulse.com) and slightly edited to change specific parameters to improve performance.

inferencing-library: Arduino library with source code including sketch file, tflite model and header files needed for deployment. 

deployment.ino : Arduino sketch file to deploy the model. Note that this script was developed by Edge Impulse and edited to extend the model inference to determine the final output of no mold vs mold and created LED light functionality.

report.pdf: Report on methodology, experiments, and reflections of the project

### Appendix

[Link](https://www.youtube.com/watch?v=LyyfgAZ16uk) to Video Presentation

[Link](https://studio.edgeimpulse.com/studio/184531/learning/keras-transfer-image/16) to Edge Impulse Project Page

