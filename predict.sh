python /model/preprocessing.py --data_dir data \
                               --file_name test

python /model/predict.py --data_dir data \
                         --model_dir model/saved_models \
                         --out_dir result \
                         --file_name test