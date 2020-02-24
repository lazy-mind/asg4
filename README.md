# asg4
## 1. Pre Processing ETL
This step is shown in the proprecessing.zip

## 2. Run the Pre Processing on the dataset
This step is shown in the fold "aws". job.py is the job's script code, and the json output file is shown as dev.json, eval.json, train.json. Also, we attached a screenshoot(called Job_result.docx) about successfully running the job.

## 3. Tensorflow model
This step is shown in the fold "model_training". We attached a screenshoot(called cnn-model_local_result.docx) about successfully training the model. 

To run the code locally, you need run using command-line tools: python -W ignore sentiment_training.py --train data/train/ --validation data/dev/ --eval data/eval/ --model_output_dir model/ --model_dir model/ --num_epoch 3

## 4. SageMaker training
This step is shown in the fold "sagemaker".
