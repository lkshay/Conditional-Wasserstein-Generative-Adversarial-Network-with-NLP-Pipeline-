#! /bin/bash

declare -A classes=( ["airplane"]=0 ["car"]=1 ["bird"]=2 ["cat"]=3 ["deer"]=4 ["dog"]=5 ["frog"]=6 ["horse"]=7 ["ship"]=8 ["truck"]=9)

var=${1}
#echo $var
echo "writing to file ... "
echo $var>"test.txt"
#cat "test.txt"
out=`./fastText/fasttext predict-prob fastText/model.bin test.txt 1 | cut -d' ' -f1`

class=${out:9}
class_num=${classes[$class]}
echo label = $class >>"test.txt"
echo class = $class_num >>"test.txt"

python generate_cifar.py --CHECKPOINT_DIR=checkpoints/gan/DATASET_cifar10/LOSS_wgan/DIST_normal/MATCH_False --CLASS=$class_num
