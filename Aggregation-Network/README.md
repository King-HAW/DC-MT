# Aggregation Network

## Training
### Visdom
We use visdom to visualize the Intermediate attention results and plot the loss values, you can open the visdom sever by specifying the `<port>`. The default port is 8097, and we use 10014 in our training process.
```bash
python -m visdom.server -p <port>
``` 

### train the aggregation network with yaml file
```bash
python train.py -cfg configs/train/se50_semi_valid0.yaml
```
The training log will be stored on `./output/logs`. If you want to train the network on all folds, you can use the `fold_train.sh` bash file.
```bash
sh fold_train.sh
```

## validation
```bash
python valid.py -cfg configs/valid/se50-subject-valid0.yaml
```
The evaluation metrics will be stored in the log file. If you want to validate the models on all folds, you can use the `fold_valid.sh` bash file.
```bash
sh fold_valid.sh
```

## Code Structure
- `train.py`, `valid.py`: the entry point for training and validation.
- `fold_train.sh`, `fold_valid.sh`: the bash files for training and validation on all folds.
- `configs/`: config files for training and feature extraction.
- `data/`: the data loader.
- `models/`: creates the networks.
- `modules/losses/`: the loss functions.
- `utils/`: define the training and the support modules.

