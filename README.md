# iMap_impl
Implicit Mapping and Positioning in Real Time implementation

Current version of the implementation contains the implicit scene neural network.
The current implementation is very basic, it takes in a 3D point in world coordinates and has two output heads - Colour and Volume Density Value.

To test the implementation of the MLP, the 3D world points were back projected from the ground truth pose. It works well on certain simple and short sequences like 
the bulldozer sequence https://drive.google.com/drive/folders/128yBriW1IG_3NJ5Rp7APSTZsJqdJdfc1
For the current tests, the experiments were performed on the TUM long office dataset.

