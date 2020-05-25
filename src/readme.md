## BNN source code

The project is distributed into 3 scripts:
- [bnn_dg.py](bnn_dg.py): Data-generation script which contains:
  - `BubbleReader`: A class to read and store individual void fraction tensors
  - `BubbleDirReader`: A class to load and segregate entire experimental datasets
  - `scalers`: A class with custom scaling for input/output used in BNN
  - `vr_nn_seqgen`: Sequence generator used during training/validation
- [bnn_nn.py](bnn_nn.py): Build/load neural network and train
- [bnn_pp.py](bnn_pp.py): Post-processing script which contains:
  - `map_error`: Custom function for mean/sigma of error
  - `prep_data`: Data-preparation for post-processing meaningfully
  - `model_mape`: Driver function to evaluate saved model

If you are training new models, I highly recommend to modify the post-processing methods to match your dataset.
