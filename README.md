# asg4
## 1. Pre Processing ETL
This step is shown in the proprecessing.zip

## 2. Run the Pre Processing on the dataset
This step is shown in the fold "aws", and the json file is shown as dev.json, eval.json, train.json

## 3. Tensorflow model
This step is shown in the fold "model_training 5.44.01 PM/model_training". To run the code locally, you need run $python -W ignore sentiment_training.py --train data/train/ --validation data/dev/ --eval data/eval/ --model_output_dir model/ --model_dir model/ --num_epoch 3

## 4. SageMaker training
This step is shown in the fold "sagemaker"
