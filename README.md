## System Requirements
### Hardware Requirements
All experiments were conducted on a computational server equipped with an Intel(R) Core(TM) i7-9700K CPU @ 3.60 GHz and an NVIDIA RTX A6000 GPU with 48GB of memory.
#### Python Version
Python 3.9 or higher is required
#### PytORCH Dependencies
````
pip install torch==1.12.1
````

## FOR PEMS Dataset
### Folder Structure
````shell
-PEMS
  -datasets 
    -PEMS08
````
### Operations
All operations must be performed in the PEMS folder after downloading [DataLink](https://drive.google.com/file/d/1VEsGcwkulDQ47ihbIcw-wEiXpHEwltJy/view) and placing the dataset.

#### Training
Example:
````shell
python train.py -c config/PEMS08.py --gpus '0'
````
## For LargeST Dataset

### Folder Structure
````shell
-LargeST
  -data
    -sd
````
### Operations
All operations must be performed in the LargeST folder after downloading [DataLink](https://drive.google.com/file/d/1GRP8ImMuyPr4n_ofKnnyI3CYgUvZrNUG/view?usp=drive_link) and placing the dataset.

#### Training
Example:
````shell
python main.py --dataset SD --mode 'train'
````

