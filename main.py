import numpy as np
from nerf import *

### SLAM Class ###
## Contains the basic placeholders for the various modules
## Sequential version. Need to parallelize after verifying the end-end pipeline

class iMap:
    def __init__():
        ## Velocity (Use Constant velocity model)
        self.mVel = np.identity(4, dtype = np.float32)
        self.Tcw = np.identity(4, dtype = np.float32)
        self.lastTcw = np.identity(4, dtype = np.float32)
        self.vKFs = []

        images, images_depth, poses = prepareLOSeq("/home/srikanths/rgbd_dataset_freiburg3_long_office_household/gt_data.txt")
        flen = 539.2 * (224/480)
        nerf = NERF(images, images_depth, poses, flen)

    def trackFrame(self, image_rgb, image_depth):
        self.lastTwc = self.Tcw.copy()
        updatedTcw = np.matmul(self.mVel, self.lastTcw)
        self.Tcw = trackWithIcp(image_rgb, image_depth, updatedTcw)
        self.mVel = np.matmul(self.Tcw, np.inverse(self.lastTcw))

        ## Decide for keyframe
        # depth threshold
        t_d = 0.1
        isKF = False
        curr_depth_pred = nerf.getCurrentDepth()
        imD = tf.convert_to_tensor(image_depth)
        p = tf.reduce_mean(tf.math.abs(imD - curr_depth_pred) / imD)
        if(p < t_d):
            isKF = true
            self.createKF(image_rgb, image_depth)
        
        ## Mapping 

    def createKF(self, image_rgb, image_depth):
        ## Implement mapping module
        pass