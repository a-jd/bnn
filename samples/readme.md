## Samples from experimental data

The tar contains samples for Test A to F (refer to Fig. 1 [arXiv:2003.08182 [physics.flu-dyn]](https://arxiv.org/abs/2003.08182)). 
Each sample's size is (512,16,16) and is saved in a MATLAB mat file format.
To load the samples it is recommended to use `scio.loadmat`.
Each sample will need to be scaled using `scalers.void_scale` in [src/bnn_dg.py](../src/bnn_dg.py).
The output will need to be descaled using `scalers.vel_descale`.
