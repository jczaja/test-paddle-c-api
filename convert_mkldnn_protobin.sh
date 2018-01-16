#!/bin/bash
python -m paddle.utils.dump_config trainer_config.py 'use_mkldnn=1' --binary > trainer_config.bin
