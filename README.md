# ConvNeXt-audio
Adapting a ConvNeXt model to audio classification on AudioSet

ConvNeXt-audio is a codebase that adapts the ConvNeXt architecture for audio classification in AudioDet. AudioDet is a large-scale dataset of human-labeled audio events, with over 2 million 10-second audio clips labeled with a vocabulary of 527 audio event classes.

ConvNeXt is a convolutional neural network architecture that was originally designed for vision tasks, but can be adapted for other domains as well. The architecture combines the strengths of two types of convolutions, namely the traditional 2D convolution and the depthwise separable convolution, and achieves state-of-the-art results on various vision classification benchmarks.

In this repository, we provide a PyTorch implementation of the ConvNeXt architecture adapted for Audioset. We provide scripts for training and evaluating ConvNeXt model variants.

## Requirements

The code has been tested on Python 3.7 and PyTorch 1.10. 

## Usage

To train a Tiny ConvNeXt model on the full AudioSet, run the following command in the scripts directory:

```bash
./launch_4.sh ConvNextTiny_pretrainedTrue_specAugm28_stemDim224_56_dp4e-1_lr4e-3_wd05_speedTrue main.py 4_train_multiGPU_20h.sbatch
```

A pre-trained model will be made available for download upon paper acceptance, together with a more informative and detailed readme to run the experiments. 


## Acknowledgements

Our implementation is based on the `ConvNeXt` codebase by Facebook Research, and on the PANN audio classification CNNs. We also thank the authors of the `Audioset` dataset for providing a valuable resource for audio research.

## License

This code is released under the MIT License. See the LICENSE file for details.
