
# coding: utf-8

# In[2]:


#%HTML
#<style> code {background-color : pink !important;} </style>


# Camera Calibration with OpenCV
# ===
# 
# ### Run the code in the cell below to extract object points and image points for camera calibration.  

# In[3]:

import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt

#get_ipython().magic('matplotlib inline')


##vvibin added
criteria = (cv2.TERM_CRITERIA_EPS + cv2. TERM_CRITERIA_MAX_ITER,30,0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*5,3), np.float32)
objp[:,:2] = np.mgrid[0:6, 0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
images = glob.glob('/home/camera-calibration/*.*')
print (images)

# Step through the list and search for chessboard corners
for idx, fname in enumerate(images):
    print ("idx=", idx ,fname)
    img = cv2.imread(fname)
    #print("origninal")
    #cv2.imwrite('img.jpg', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #gray = cv2.resize(gray,None,fx=0.5,fy=0.5)
    #clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    #gray = clahe.apply(gray)
    
    #print("afer clahe")
    #cv2.imwrite('gray.jpg',gray)
    
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (6,5),0)

    #x" , mtx ,"new matrix",newmtx)
    #If found, add object points, image points
    if ret == True:
        
        print ("found chess board",fname)
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix( gray, corners, (11,11), (-1,-1), criteria )

	#vvibin ended
        imgpoints.append(corners2)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (7,7), corners, ret)
        #write_name = 'corners_found'+str(idx)+'.jpg'
        #cv2.imwrite(write_name, img)
        #cv2.imshow('img', img)
        #cv2.waitKey(500)

#cv2.destroyAllWindows()


# In[3]:

### If the above cell ran sucessfully, you should now have `objpoints` and `imgpoints` needed for camera calibration.  Run the cell below to calibrate, calculate distortion coefficients, and test undistortion on an image!


# In[ ]:

import numpy as np
import cv2
import glob
#import matplotlib.pyplot as plt
import pickle
#get_ipython().magic('matplotlib inline')

# Test undistortion on an image
img = cv2.imread('/home/camera-calibration/test_image.jpg')
img_size = (img.shape[1], img.shape[0])
print ( "size" , img_size )

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,gray.shape[::-1],None,None)

#print ("old matrix" , mtx ,"new matrix",newmtx)

# vvibin added
h,w = img.shape[:2]
newmtx, roi = cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
# vvibin ended

print ("old matrix" , mtx ,"new matrix",newmtx ,"roi",roi)

dst = cv2.undistort(img, mtx, dist, None, mtx)
# vvibin added
#x,y,w,h = roi
#dst = dst[x:y, w:h]
#print ("roi" ,roi ,"dst",dst)

#vvibin ended

cv2.imwrite('/home/camera-calibration/test_undist.jpg',dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera-calibration/wide_dist_pickle.p", "wb" ) )
#dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
# Visualize undistortion
#f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
#ax1.imshow(img)
#ax1.set_title('Original Image', fontsize=30)
#ax2.imshow(dst)
#ax2.set_title('Undistorted Image', fontsize=30)


# In[ ]:




# In[ ]:




# In[ ]:



