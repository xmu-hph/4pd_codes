for var in en; do
  curl --location --request POST 'http://contest.4pd.io:8080/submit' \
    --header 'Authorization: Bearer a20bec9873586129d1384ba155c0daeb' \
    --form-string "benchmark=tts_stream_${var}" \
    --form-string 'contributors=hupenghui,liyihao,wangzhangcheng' \
    --form-string 'source_code=https://gitlab.4pd.io/hupenghui/automl_cad_related' \
    --form-string 'description=basemodel:tired' \
    --form 'config_file=@"/home/mnt/4pd_codes/tts/config.yaml"'
done

curl --location --request POST 'http://contest.4pd.io:8080/submit' \
  --header 'Authorization: Bearer a20bec9873586129d1384ba155c0daeb' \
  --form-string "benchmark=voice_clone" \
  --form-string 'contributors=hupenghui,liyihao,wangzhangcheng' \
  --form-string 'source_code=https://gitlab.4pd.io/hupenghui/automl_cad_related' \
  --form-string 'description=basemodel:tired' \
  --form 'config_file=@"/home/mnt/4pd_codes/tts/config.yaml"'