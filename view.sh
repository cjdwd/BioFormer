#!/bin/bash
n="1000"
if [ "$1" == "kd" ]
then
logfile="./logs/kd.out"
tail -f -n $n ${logfile}
fi

if [ "$1" == "kiba" ]
then
logfile="./logs/kiba.out"
tail -f -n $n ${logfile}
fi

if [ "$1" == "dude" ]
then
logfile="./logs/dude.out"
tail -f -n $n ${logfile}
fi


