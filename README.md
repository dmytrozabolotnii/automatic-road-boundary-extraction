# Automatic Road Boundaries Extraction for High Definition maps project

## Description

This project contains all created scripts for the data pre-processing, Stage 1 and Stage 2 of the Automatic Road Boundaries Extraction for HighDefinition maps paper by Dmytro Z. et.al. [future pre-print link]

Provided scripts work exclusively using the publicly available [Nuscenes](https://www.nuscenes.org/nuscenes) main dataset.

## Requirements

- Python 3.7
- Nuscenes devkit
- Keras GPU with Tensorflow backend
- Shapely
- SciPy and Skimage
- FilFinder
- tqdm

### Instruction

- Use Unix, Windows compatibility is not guaranteed without some code changes
- Install required Python modules  
- Copy data folder structure from example folder to base level folder  
- Nuscenes data set by default goes into ../data/sets/nuscenes/  
- List of the runnable Python scripts and their functions is below

The general loop to reproduce results:
1. Run dataset_generators/mass_stage1_input_generation.py to generate BEV image dataset from Nuscenes dataset
2. Run stage1/stage1_train.py to train Stage 1 neural network on created dataset
3. Run stage1/stage1_fullsizeeval.py to generate intermediate Stage 1 output for Stage 2 future processing
4. Run stage2/stage2_morp.py to generate Stage 2 output 

## Scripts functionality

Every runnable script has a multitude of parameters that can be explained with --help. Please see code comments for more implementation details.

### Dataset generators

*stage1_input_generation.py*: Generates the training/testing BEV images (both input and output) from Nuscenes data single scene for the purpose of Stage 1 network training/evaluation. Optionally generates random image croppings for the data augmentation. Due to the output images generation process, automatic image cropping provided by the frameworks is non-accurate.

*mass_stage1_input_generation.py*: Script for mass running the previous generator to generate BEV images from multiple scenes of Nuscenes main trainval dataset at the same time using same parameters specified in the parameters of this script.

### Stage1

*stage1_train.py*: Generates and trains Stage 1 neural network from the generated dataset in "input" folder according to the configuration set mainly in the parameters. Requires creating datafile in the dataset folder containing all names of the scenes in the training dataset (see example folder). Can be used to read already pre-trained weights and output predictions for given dataset, however resizes images to the given shape, runs them through network, then resizes back to the original size.

*stage1_fullsizeeval*: Loads pre-trained Stage 1 neural network weights and processes a single scene image using original proportions and outputs to the given output folder.

### Stage2

*stage2_morp.py*: Reads the output of Stage 1 neural network and applies necessary morphological operations for the refinement of the road boundaries. Requires creating datafile in the Stage 1 output folder containing all names of the scenes that needs to be processed. By specifying ground truth folder, combines input data, ground truth data and Stage 2 refined boundaries into demo images. Calculates confusion matrix metrics for the given refined Stage 1 output and the original ground truth.
