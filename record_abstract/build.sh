#!/bin/bash
set -Ceu
imgroot="harbor.4pd.io/lab-platform/pk_platform/model_services/basemodel_ability_hupenghui:"
imgtag="german"
#tts_clone
#tts_stream
#tts_standard
imgname=$imgroot$imgtag
docker build -t $imgname .
docker push $imgname