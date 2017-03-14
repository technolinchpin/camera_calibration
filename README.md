## Camera Calibration with OpenCV

The IPython notebook in this repository contains code to calculate the camera matrix and distortion coefficients using the images in the "calibration_wide" folder. [This is the fork from https://github.com/udacity/CarND-Camera-Calibration ]

### Convert Ipython notebook to python module
The IPython module sometime crash the Jupyter kernel. Hence the python notebook is converted to python module using Ipython nbconvert option [https://ipython.org/ipython-doc/3/notebook/nbconvert.html]

To work with camera_calibration.py
  1. Add a new folder camera_calibrate and place the checkboard pattern files.
  2. pyton camera_calibration.py
  3. If everything works well the camera distortion correction coefficients are saved in  wide_dist_pickle.p
 
the distorition corrected image for the test_image.jpg will be saved as test_undist.jpg

