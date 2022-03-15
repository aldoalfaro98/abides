#!/bin/bash

seed=123456789
ticker=ABM
date=20200101
book_freq=30T
log=simple_spoofing_${seed}

# (1) Sparse Mean Reverting Oracle Fundamental
#python -u abides.py -c market_manipulation -t ${ticker} -d ${date} -l ${log} -s ${seed} -e

# (2) External Fundamental file
fundamental_path=${PWD}/data/synthetic_fundamental.bz2
python -u abides.py -c market_manipulation -b${book_freq} -t ${ticker} -d ${date} -f ${fundamental_path} -l ${log} -s ${seed} -e

#Plotting
cd util/plotting && python -u liquidity_telemetry.py ../../log/simple_spoofing_123456789/EXCHANGE_AGENT.bz2 ../../log/simple_spoofing_123456789/ORDERBOOK_ABM_FULL.bz2 \
-o simple_spoofing.png -c configs/plot_09.30_11.30.json && cd ../../
