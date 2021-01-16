#!/bin/bash

if [ $# -ne 6 ]
then
    echo 'Usage: ./run_all.sh <training_set> <training_labels> <endpoints> <fenemes> <test_set> <test_output>'

else
    python3 endpoints.py $1 $3 $4

    python3 asr.py $1 $2 $4 $3

    python3 predict.py $5 $6

fi
