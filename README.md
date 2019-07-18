# Chord progression modeling

## Installation

This section presents instructions to install necessary packages using [Anaconda Platform](https://www.anaconda.com/distribution/).

```
conda create -n cpm python=3.7 -y
conda activate cpm
pip install allennlp
conda install pytorch torchvision cudatoolkit=9.0 -c pytorch -y
```

## Usage
### Generate datasets for training, validation and testing (Optional)

Run to regenerate datasets:
```
python generate datasets.py
```

### Training

Run:
```
python main.py
```

This will log results in `logs` folder.
