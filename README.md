# Chord progression modeling with progressive training

## Installation

This section presents instructions to install necessary packages using [Anaconda Platform](https://www.anaconda.com/distribution/).

```
conda create -n cpm python=3.7 -y
conda activate cpm
pip install allennlp
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch -y
```

## Usage
### Generate datasets (Optional)

Run to regenerate datasets:
```
python generate datasets.py
```

### Training

Run:
```
python new_main.py
```

This will log results in `logs` folder.  
The file `logs.json` is the log database. To convert this file into `.xlsx`, run `python log_to_excel.py`.

### Configurations

Each function in file `new_main.py` was used to run an experiment in our paper.  
To run a new experiment, please create a new dictionary `hparams` and pass it into function `train_and_evaluate().`  
For a default `hparams`, please call the function `get_default_hparams()`.  
There are 3 `training_mode`: `one_hot`, `decreased_temperature`, and `fixed_temperature`.

### Progressive training for image classification

We also provide progressive training framework for image classification with Cifar10 at this [repo](https://github.com/kienvu58/image_classification).  
The code for image classification was refactored, thus it is easier to read and extend than the code from this repo.

### Contact

If you have any question, feel free to create an issue or contact me by email: [vtk5995@gmail.com](mailto:vtk5995@gmail.com).
