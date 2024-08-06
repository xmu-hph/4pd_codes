#!/bin/bash
set -Ceu
imgroot="harbor-contest.4pd.io/hupenghui/tts:"
imgtag="tts_stream_clone_1002"
#tts_stream_clone_2012
#tts_stream_clone_1655
#tts_stream_clone_without_zero_res
imgname=$imgroot$imgtag
docker build -t $imgname .
docker push $imgname