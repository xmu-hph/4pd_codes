for var in en cn; do
  curl --location --request POST 'http://contest.4pd.io:8080/submit' \
    --header 'Authorization: Bearer a20bec9873586129d1384ba155c0daeb' \
    --form-string "benchmark=tts_stream_${var}" \
    --form-string 'contributors=hupenghui,liyihao,wangzhangcheng' \
    --form-string 'source_code=https://gitlab.4pd.io/hupenghui/automl_cad_related' \
    --form-string 'description=basemodel:stream_cn' \
    --form 'config_file=@"config.yaml"'
done
#a20bec9873586129d1384ba155c0daeb
#e4849e263e914472
#voice_clone
#tts_ja
#tts_stream_${var}
#ar tr es ja pt fr de it nl pl en
#cn ru ar tr es pt ja fr de it nl pl
#cn ar es pt ja fr
#         40长    80长
# 指指定gpu  1       1
# 不指定gpu  1       1
# cpu       1       1