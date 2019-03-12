#!/bin/bash

for i in {0..2}; do
    for a in {RNN_LSTM,RNN_LSTM_MoO,RNN_LSTM_MoW,RNN_LSTM_MoO_time,RNN_ILSTM};
    do
	CUDA_VISIBLE_DEVICES=6 python main.py -m $a &
    done
    wait
done
