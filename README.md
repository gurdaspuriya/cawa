# cawa
Credit Attribution with Attention

The folder Code contains the python code for Credit Attribution With Attention (CAWA) and other helper scripts for evaluation. The code uses Pytorch framework.
The folder Data contains the preprocessed data for the five text datasets: Movies, Ohsumed, TMC2007, Patents and Delicious.

usage: python Code/cawa.py <arguments>

Following arguments can be used

  -d DATAPATH, --datapath DATAPATH: Path to the folder containing data files.
                        
  -c CLASSES, --classes CLASSES: Number of classes.
                        
  -s SEED, --seed SEED: Seed for random initializations.
  
  -a ALPHA, --alpha ALPHA: Alpha (Default 0.2).
  
  -k KERNEL_SIZE, --kernel_size KERNEL_SIZE: Kernel size for smoothing
  
  -v STANDARD_DEVIATION, --standard_deviation STANDARD_DEVIATION: Standard deviation for the gaussian kernel, negative input means simple averaging
  
  -l LEARNING, --learning LEARNING: Learning rate (Default 0.001).
  
  -y NODES, --nodes NODES: Number of nodes in neural network (Default 256).
  
  -e EPOCH, --epoch EPOCH: Num epochs (Default 100).
  
  -b BATCH, --batch BATCH: Batch size (Default 256).
  
  -p DROPOUT, --dropout DROPOUT: Dropout probability (Default 0.5).
  
  -u UNUSED, --unused UNUSED: Use null class (Default 0).
  
  -m CLIPPING, --clipping CLIPPING: Clipping value (Default 0.25).
  
  -f CHECK, --check CHECK: Check flag (Default 10), write results to the file after every <these> epochs.
  
  -q SCRIPTS, --scripts SCRIPTS: Path to the folder containing python scripts.
  
  -r RESULTS, --results RESULTS: Path to the results output file.

Example usage for the Movies dataset:

python Code/cawa.py --datapath Data/cmumovies --classes 6 --seed 0 --alpha 0.2 --kernel_size 3 --standard_deviation -1 --learning 0.001 --nodes 256 --epoch 100 --batch 256 --dropout 0.5 --unused 0 --clipping 0.25 --check 10 --scripts Code/scripts --results results.txt

The output will be as follows:

After every check_flag=10 epochs, the model will write the evaluation results for the credit attribution as well as multilabel classification for different values of beta to the results file. The evaluation will be performed for both the test and validation datasets.

Each line in the results file will have comma separated 24 fields as follows:
1) random seed
2) alpha
3) kernel_size
4) kernel_sd
5) learning_rate
6) hidden_dim
7) epoch
8) batch_size
9) dropout
10) use_null
11) clipping_value
12) beta
13) roc
14) roc_macro
15) micro_f1
16) samples_f1
17) macro_f1
18) weighted_f1
19) sov_strict_valid
20) sov_smooth_valid
21) accuracy_valid
22) sov_strict_test
23) sov_smooth_test
24) accuracy_test


The fields 13 to 18 correspond to the evaluation on multilabel classification on the test dataset. The fields 19 to 21 correspond to the evaluation of credit attribution on the validation set and the fields 22 to 24 correspond to the evaluation on test set.
For multilabel classification, the metrics of interest are the fields 13, 14 and 16.
For credit attribution, the metrics of interest are the fields 23 and 24.

The best hyperparameter values for different datasets are:
Dataset    α   β
Movies    0.2 0.1
Ohsumed   0.1 0.1
TMC2007   0.1 0.3
Patents   0.5 0.3
Delicious 0.1 0.2
