## To RUN 
python blur.py -i ./blur_images -s "RTC:blur/blur.py/result.json"
--sagemaker

python process.py -i images -s result.json
 

 python process.py -i './pic' -s "result.json"
