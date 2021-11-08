# iMap_impl
Implicit Mapping and Positioning in Real Time implementation

Current version of the implementation contains the implicit scene neural network.
The current implementation is very basic, it takes in a 3D point in world coordinates and has two output heads - Colour and Volume Density Value.

To test the implementation of the MLP, the 3D world points were back projected from the ground truth pose. It works well on certain simple and short sequences like 
the bulldozer sequence https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
For the current tests, the experiments were performed on the TUM long office dataset.

Sample Input View:<br />
![alt text](https://github.com/srikanths001/iMap_impl/blob/main/images/input.png)
Network rendered image for an unseen view: <br />
![alt text](https://github.com/srikanths001/iMap_impl/blob/main/images/plots_tum_res.png)
<br />

Some of the optimizations mentioned in the paper to be tried out are, <br />

TODO:
- Although the current implementation uses positional embedding, I have not yet implemented the optimization of the embedding matrix B.
- Camera Tracking Module: The paper does not mention what approach is being used for the tracking module. I am assuming they have used a variation of ICP algorithm for computing the pose of the new frame. This has to be implemented.
- Joint optimization of the pose of the keyframes. The paper does not explicitly mention the details of the pose optimization. I assume some kind of non-linear least squares is required.
- Keyframe and image pixels sampling. The paper mentions that to make the whole system real time, certain number of points are sampled and fed to the network. This has to be implemented. Currently the system uses all the pixels in the image for backprojection.
- Include keyframes for optimization in mapping module. Implement mapping module. Also, project 3D points from the last M keyframes during joint optimization. Currently only the current frame's points are projected and optimized.
- Parallelize the tracking and mapping thread. The implementation can be in c++.
