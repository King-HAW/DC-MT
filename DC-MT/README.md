# DC-MT

## Training
### Visdom
We use visdom to visualize the Intermediate attention results and plot the loss values, you can open the visdom sever by specifying the `<port>`. The default port is 8097, and we use 10014 in our training process.
```bash
python -m visdom.server -p <port>
``` 

### train the DC-MT with yaml file
```bash
python train.py -cfg configs/train/se50_semi_valid0.yaml
```
The training log and visualization results will be stored on `./output/logs` and `./output/train` respectively. If you want to train the network on all folds, you can use the `fold_train.sh` bash file.
```bash
sh fold_train.sh
```

## Feature Extraction
```bash
python feature_extract.py -cfg configs/feature_extract/se50-subject-valid0.yaml
```

The extracted feature will be saved in `./feature` folder. If you want to extract features on all folds, you can use the `fold_feature_extract.sh` bash file.
```bash
sh fold_feature_extract.sh
```

After all features are extracted, using the following command to copy features from current folder to `../Aggregation-Network` folder.
```bash
cp -r ./feature ../Aggregation-Network
```

## Code Structure
- `train.py`, `feature_extract.py`: the entry point for training and feature extraction.
- `fold_train.sh`, `fold_feature_extract.sh`: the bash files for training and feature extraction on all folds.
- `configs/`: config files for training and feature extraction.
- `data/`: the data loader.
- `models/`: creates the networks.
- `modules/attentions/`: the proposed foreground attention module.
- `modules/losses/`: the loss functions.
- `utils/`: define the training and feature extraction process and the support modules.