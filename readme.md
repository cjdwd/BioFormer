#FPN BiLSTM
Efficiently predicting Drug Target Interaction
***
###Datasets
In the data folder, we provide all three 
processed datasets used in our model: 
BindingDB_Kd, KIBA, and DUDE. Besides drug smiles
and protein sequence, we provide protein pdb
as well as contactmap.
***
###Run
To train on kd, kiba or dude, simply run
bash train.sh [dataset], for example, kd
```shell
bash train.sh kd
```
To view the logs while training, run
```shell
bash view.sh kd
```


