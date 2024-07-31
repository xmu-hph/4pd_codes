curl --location --request POST 'http://contest.4pd.io:8080/submit' \
  --header 'Authorization: Bearer a20bec9873586129d1384ba155c0daeb' \
  --form-string "benchmark=full_text_summarize_chinese" \
  --form-string 'contributors=hupenghui,liyihao,wangzhangcheng' \
  --form-string 'source_code=https://gitlab.4pd.io/hupenghui/automl_cad_related' \
  --form-string 'description=basemodel:tired' \
  --form 'config_file=@"config.yaml"'