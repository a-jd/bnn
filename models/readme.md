## Pre-trained weights for Keras models

Model A to E weights trained with full experimental dataset.
Refer to Fig. 2 [arXiv:2003.08182 [physics.flu-dyn]](https://arxiv.org/abs/2003.08182) for each model's architecture.
Load with `keras.models.load_model`. 
The input for each model will be a tensor of shape (512,16,16,1) and the output will be an array of shape (2).
To accommodate different input sizes, the neural network will require restructuring and re-training. 
