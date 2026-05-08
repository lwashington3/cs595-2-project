#!/bin/bash

for MODEL in $(ssh cs595 find .cache/huggingface/hub -type d -regextype posix-egrep -regex ".cache/huggingface/hub/models--.*/snapshots/.*/pretrained_.*"); do
	mkdir -p "$MODEL";
	ssh cs595 tar -czvf - "$MODEL" | tqdm --bytes --desc="Downloading" | tar -xzvf - -C "$MODEL";
done | tqdm --desc="Downloading Pretrained models"

mkdir -p ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/pretrained_cnn_dailymail
scp -prv cs595:~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/pretrained_cnn_dailymail ~/.cache/huggingface/hub/models--google--gemma-3-1b-it/snapshots/dcc83ea841ab6100d6b47a070329e1ba4cf78752/pretrained_cnn_dailymail

scp cs595:LTH.db ./

mkdir ../evaluation
scp -prv cs595:evaluation ./evaluation

mkdir ../models
scp -prv cs595:models ./models

