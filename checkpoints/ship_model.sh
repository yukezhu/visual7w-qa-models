DATE=$(date +"%Y%m%d%H%M")
SUFFIX=$1
echo 'shipping model_id'${SUFFIX} ${DATE}
mv model_id${SUFFIX}.json model_zoo/model_id${SUFFIX}_${DATE}.json
mv model_id${SUFFIX}.t7 model_zoo/model_id${SUFFIX}_${DATE}.t7
