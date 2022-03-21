#!/bin/bash
trainfile="train.py"
if [ "$1" == "kd" ]
then
logfile="./logs/kd.out"
fi

if [ "$1" == "kiba" ]
then
logfile="./logs/kiba.out"
fi

if [ "$1" == "dude" ]
then
logfile="./logs/dude.out"
fi

nohup python -u ${trainfile} $1 > ${logfile} 2>&1 &
