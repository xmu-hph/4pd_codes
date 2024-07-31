#!/bin/bash
set -Ceu
imgroot="harbor-contest.4pd.io/hupenghui/summarize:"
imgtag="qwen-spacy"
#tts_clone
#tts_stream
#tts_standard
imgname=$imgroot$imgtag
docker build -t $imgname .
docker push $imgname