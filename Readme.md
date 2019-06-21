## VDSR impletation with keras
### Author's page : https://cv.snu.ac.kr/research/VDSR/

## How to use
#### Train command
> python train.py --batchsize 'NUMBER' --epoch 'NUMBER' --train 'RELATIVE PATH TO TRAIN DATA' --valid 'RELATIVE PATH TO VALID DATA' --expansion 'EXPANSION OF DATA'

##### for example,
> python train.py --batchsize 4 --epoch 1000 --train ./TrainData --valid ./TestData --expansion png


#### Test command
> python test.py --test 'RELATIVE PATH TO TEST DATA' --network 'NETWORK NAME' --expansion 'EXPANSION OF TEST DATA'

##### for example,
> python test.py --test ./TestData --network VDSR_itr1000_bs4_TrainData_2019-06-21-18-28.hdf5 --expansion png

## Versions
#### cuda 9.0
#### python 3.6.8
#### tensorflow-gpu 1.12.0
#### keras 2.2.4
