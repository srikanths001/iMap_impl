# iMap_impl
Implicit Mapping and Positioning in Real Time implementation

Current version of the implementation contains the implicit scene neural network.
The current implementation is very basic, it takes in a 3D point in world coordinates and has two output heads - Colour and Volume Density Value.

To test the implementation of the MLP, the 3D world points were back projected from the ground truth pose. It works well on certain simple and short sequences like 
