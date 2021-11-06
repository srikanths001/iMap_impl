import tensorflow as tf
import numpy as np

tf.compat.v1.enable_eager_execution()
import matplotlib.pyplot as plt
import pdb

import cv2
from scipy.spatial.transform import Rotation as R

gt_path = "/home/kaushikdas/srikanths/rgbd_dataset_freiburg3_long_office_household/"

def posenc(x, L_embed):
    rets = [x]
    for i in range(L_embed):
      for fn in [tf.sin, tf.cos]:
        rets.append(fn(2.**i * x))
    return tf.concat(rets, -1)

def init_model(L_embed, D=8, W=256):
    relu = tf.keras.layers.ReLU()    
    dense = lambda W=W, act=relu : tf.keras.layers.Dense(W, activation=act)
    # pdb.set_trace()

    inputs = tf.keras.Input(shape=(3 + 3*2*L_embed)) 
    outputs = inputs
    for i in range(D):
        outputs = dense()(outputs)
        if i%4==0 and i>0:
            outputs = tf.concat([outputs, inputs], -1)
    outputs = dense(4, act=None)(outputs)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_rays(H, W, focal, c2w):
    i, jc = tf.meshgrid(tf.range(W, dtype=tf.float32), tf.range(H, dtype=tf.float32), indexing='xy')
    dirs = tf.stack([(i-W*.5)/focal, -(jc-H*.5)/focal, -tf.ones_like(i)], -1)
    rays_d = tf.reduce_sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)
    rays_o = tf.broadcast_to(c2w[:3,-1], tf.shape(rays_d))
    return rays_o, rays_d

def render_rays(network_fn, rays_o, rays_d, near, far, N_samples, L_embed, rand=False):

    def batchify(fn, chunk=1024*32):
        return lambda inputs : tf.concat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    
    # Compute 3D query points
    z_vals = tf.linspace(near, far, N_samples)
    if rand:
      z_vals += tf.random.uniform(list(rays_o.shape[:-1]) + [N_samples]) * (far-near)/N_samples
    pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    
    # Run network
    pts_flat = tf.reshape(pts, [-1,3])
    pts_flat = posenc(pts_flat, L_embed)
    raw = batchify(network_fn)(pts_flat)
    raw = tf.reshape(raw, list(pts.shape[:-1]) + [4])
    
    # Compute opacities and colors
    sigma_a = tf.nn.relu(raw[...,3])
    rgb = tf.math.sigmoid(raw[...,:3]) 
    
    # Do volume rendering
    dists = tf.concat([z_vals[..., 1:] - z_vals[..., :-1], tf.broadcast_to([1e10], z_vals[...,:1].shape)], -1) 
    alpha = 1.-tf.exp(-sigma_a * dists)  
    weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    
    rgb_map = tf.reduce_sum(weights[...,None] * rgb, -2) 
    depth_map = tf.reduce_sum(weights * z_vals, -1) 
    acc_map = tf.reduce_sum(weights, -1)

    dm = tf.expand_dims(depth_map, axis=2)
    dvar = tf.reduce_sum(weights * (dm - z_vals) * (dm - z_vals), -1)

    return rgb_map, depth_map, acc_map, dvar

trans_t = lambda t : tf.convert_to_tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1],
], dtype=tf.float32)

rot_phi = lambda phi : tf.convert_to_tensor([
    [1,0,0,0],
    [0,tf.cos(phi),-tf.sin(phi),0],
    [0,tf.sin(phi), tf.cos(phi),0],
    [0,0,0,1],
], dtype=tf.float32)

rot_theta = lambda th : tf.convert_to_tensor([
    [tf.cos(th),0,-tf.sin(th),0],
    [0,1,0,0],
    [tf.sin(th),0, tf.cos(th),0],
    [0,0,0,1],
], dtype=tf.float32)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]]) @ c2w
    return c2w

class NERF:
    def __init__(self, input_images, images_depth, poses, focal_len, N_samples = 64, N_iters = 1000, L_embed = 6, i_plot = 25):
        ## Input images
        self.images_rgb = input_images
        self.images_depth = images_depth
        self.imPoses = poses
        self.focal_len = focal_len
        self.H = self.images_rgb.shape[1]
        self.W = self.images_rgb.shape[2]
        ## Number of embedding levels
        self.L_embed = L_embed
        ## Number of depth samples
        self.N_samples = 64
        ## Number of iterations
        self.N_iters = 1000
        ## Init Model
        self.model = init_model(L_embed = self.L_embed)
        ## Debug info
        self.i_plot = 25
        self.psnrs = []
        self.iternums = []
        self.curr_depth = None
    
    def getCurrentDepth():
        return self.curr_depth
    
    def getModel():
        return self.model

    def train_model(self):
        print(self.model.summary())
        optimizer = tf.keras.optimizers.Adam(5e-4)

        psnrs = []
        iternums = []
        min_loss = None

        import time
        ct = time.time()
        #pdb.set_trace()
        for i in range(self.N_iters+1):
            img_i = np.random.randint(self.images_rgb.shape[0])
            target = self.images_rgb[img_i]
            target_depth = self.images_depth[img_i]
            pose = self.imPoses[img_i]
            rays_o, rays_d = get_rays(self.H, self.W, self.focal_len, pose)
            with tf.GradientTape() as tape:
                rgb, depth, acc, depth_var = render_rays(self.model, rays_o, rays_d, near=0.5, far=3.0, N_samples=self.N_samples, L_embed=self.L_embed, rand=True)
                #loss = tf.reduce_mean(tf.square(rgb - target))
                loss_p = tf.reduce_sum(tf.math.abs(rgb - target))
                loss_g = tf.reduce_sum(tf.math.abs(depth - target_depth) / tf.math.sqrt(depth_var))
                self.curr_depth = depth.copy()
                ## Photometric loss scaling factor
                lambda_p = 5
                loss = loss_g + lambda_p * loss_p
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            if(min_loss == None):
                min_loss = loss
            elif(loss < min_loss):
                min_loss = loss
                print("Save model state as current best loss achieved {}".format(loss))
                self.model.save_weights("weights_net_it_" + str(i) + ".hdf5")
            print("It: {}".format(i))

            if i%self.i_plot==0:
                # Render the holdout view for logging
                # pdb.set_trace()
                rays_o, rays_d = get_rays(self.H, self.W, self.focal_len, pose)
                rgb, depth, acc, depth_var = render_rays(self.model, rays_o, rays_d, near=0.5, far=3.0, N_samples=self.N_samples, L_embed=self.L_embed)
                #loss = tf.reduce_mean(tf.square(rgb - target))
                loss_p = tf.reduce_sum(tf.math.abs(rgb - target))
                loss_g = tf.reduce_sum(tf.math.abs(depth - target_depth) / tf.math.sqrt(depth_var))
                loss = loss_g + lambda_p * loss_p
                psnr = -10. * tf.math.log(loss) / tf.math.log(10.)
                print("i: {}, Loss: {}, psnr: {}, time: {}, mean depth: {}".format(i, loss, psnr, (time.time() - ct) / self.i_plot, np.mean(depth)))
                ct = time.time()

                psnrs.append(psnr.numpy())
                iternums.append(i)
        
        plt.figure(figsize=(10,4))
        plt.subplot(121)
        plt.imshow(rgb)
        plt.title(f'Iteration: {N_iters}')
        plt.subplot(122)
        plt.plot(iternums, psnrs)
        plt.title('PSNR')
        # plt.show()
        plt.savefig("/tmp/plots/plots_tum_pe.png")
        self.model.save_weights("weights_net_tum_pe.hdf5")


def prepareLOSeq(data_path, sample_frame = 1):
    fl = open(data_path, 'r')
    lines = fl.readlines()
    rgb_files = []
    depth_files = []
    images = np.zeros((0,224,224,3), np.float32)
    images_depth = np.zeros((0,224,224), np.float32)
    poses = np.zeros((0,4,4), np.float32)
    for idx in range(0, 100, sample_frame):
        cline = lines[idx].strip("\n")
        cline = cline.split(" ")
        rgb_files.append(cline[1])
        depth_files.append(cline[3])
        cpose = np.zeros((4,4), np.float32)
        cpose[0,3] = float(cline[5])
        cpose[1,3] = float(cline[6])
        cpose[2,3] = float(cline[7])
        cpose[3,3] = 1.0
        #print("xyz: {}, {}, {}".format(cline[5], cline[6], cline[7]))
        r = R.from_quat([float(cline[8]), float(cline[9]), float(cline[10]), float(cline[11])])
        cpose[:3,:3] = r.as_matrix()
        cpose = np.linalg.inv(cpose)
        cpose = np.expand_dims(cpose, axis=0)
        poses = np.concatenate((poses, cpose), axis=0)
        im = cv2.imread(gt_path + cline[1])
        im_copy = im[:,80:im.shape[1]-80]
        im_copy = cv2.resize(im_copy, (224,224))
        im_copy = im_copy.astype(np.float32)
        im_copy /= 255.0
        im_copy = np.expand_dims(im_copy, axis=0)
        images = np.concatenate((images, im_copy), axis=0)

        im_d = cv2.imread(gt_path + cline[3], cv2.IMREAD_UNCHANGED)
        imd_copy = im_d[:,80:im.shape[1]-80]
        imd_copy = cv2.resize(imd_copy, (224,224))
        imd_copy = imd_copy.astype(np.float32)
        imd_copy /= 5000
        imd_copy = np.expand_dims(imd_copy, axis=0)
        images_depth = np.concatenate((images_depth, imd_copy), axis=0)
        # pdb.set_trace()
    #pdb.set_trace()
    return images, images_depth, poses

if __name__=='__main__':
    images, images_depth, poses = prepareLOSeq("/home/kaushikdas/srikanths/rgbd_dataset_freiburg3_long_office_household/gt_data.txt")
    images_test, images_depth_test, poses_test = prepareLOSeq("/home/kaushikdas/srikanths/rgbd_dataset_freiburg3_long_office_household/gt_data.txt", 1)
    # #pdb.set_trace()
    # print("start")
    # L_embed = 6
    # embed_fn = posenc
    # model = init_model()

    # focal = 539.2
    # focal = focal * (224/480)
    # H = 224
    # W = 224
    # #pdb.set_trace()
    # train_model()
    #pdb.set_trace()
    flen = 539.2 * (224/480)

    nerf = NERF(images, images_depth, poses, flen)
    nerf.train_model()
    pdb.set_trace()

    '''model.load_weights('weights_net_tum.hdf5')
    print("Loaded model")
    frames = []
    N_samples = 64
    pdb.set_trace()
    for th in range(0,images_test.shape[0]):
        rays_o, rays_d = get_rays(H, W, focal, poses_test[th,:3,:4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))
        if(len(frames) % 10 == 0):
            print("Completed {} frames".format(len(frames)))
    
    import imageio
    f = 'video_tum.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)'''


'''if __name__ == '__main__':
    print("start")
    L_embed = 6
    embed_fn = posenc
    model = init_model()

    data = np.load('tiny_nerf_data.npz')
    images = data['images']
    poses = data['poses']
    focal = data['focal']
    H, W = images.shape[1:3]
    print(images.shape, poses.shape, focal)

    testimg, testpose = images[101], poses[101]
    images = images[:100,...,:3]
    poses = poses[:100]

    #train_model()
    #pdb.set_trace()

    model.load_weights('weights_net.hdf5')
    print("Loaded model")
    frames = []
    N_samples = 64
    pdb.set_trace()
    for th in np.linspace(0., 360., 120, endpoint=False):
        c2w = pose_spherical(th, -30., 4.)
        rays_o, rays_d = get_rays(H, W, focal, c2w[:3,:4])
        rgb, depth, acc = render_rays(model, rays_o, rays_d, near=2., far=6., N_samples=N_samples)
        frames.append((255*np.clip(rgb,0,1)).astype(np.uint8))
        if(len(frames) % 10 == 0):
            print("Completed {} frames".format(len(frames)))
    
    import imageio
    f = 'video.mp4'
    imageio.mimwrite(f, frames, fps=30, quality=7)'''
