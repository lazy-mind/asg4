# asg4
## 1. Pre Processing ETL
Solution to step 1 is in preprocess.zip.

(I) upzip preprocess.zip, find preprocess.py in the directory

(II) the modified code is in the function: "replace_token_with_index"

## 2. Run the Pre Processing on the dataset
Solution to step 2 is in the folder "aws". 

(I) file structure: we have setup 3 folders for input data files, 3 folders for features generated from files

(II) ETL Job script: *train_job.py* is the job's script, *dev.json, eval.json, train.json* are feature files (renamed)

(III) Screenshoot for jobs, json files are in 'Job_result.docx'.

## 3. Tensorflow model
Solution to step 3 is in the folder "model_training".

(I) file structure: "data/" directory: include feature json file. "dict/" directory: include dictionary for word embedding. "model/" directory: include saved model. Others are scripts for running the machine learning task.

(II) how to run (local): 

navigate to the directory where sentiment_training.py resides

fetch a dictionary file and store in dict/. the dictionary can be found at: https://asg4.s3.us-east-2.amazonaws.com/dict/glove_vector.txt

in "training_config.json" make sure the "cloud" is assigned 0

run the script: python -W ignore sentiment_training.py --train data/train/ --validation data/dev/ --eval data/eval/ --model_output_dir model/ --model_dir model/ --num_epoch 10

(III) results: model is saved at "model"

## 4. SageMaker training
Solution to step 4 is in the folder "sagemaker"

(I) report_4.doc: includes screenshot of the notebook

(II) codes: in the folder "model_training" (outside sagemaker, the same folder as in step 3)

(III) changes: use sagemaker session to connect with S3 bucket, changes are made in:

"sentiment_dataset.py" (load feature files from s3 directory)

"sentiment_model_cnn.py" (load dictionary from s3 directory)

"training_config.json" (add parameter for connection)

(IV) how to run: 

in "training_config.json" make sure the "cloud" is assigned 1

put it to the sagemaker notebook

open sagemaker notebook terminal and run: python -W ignore sentiment_training.py --train train_features --validation dev_features --eval eval_features --model_output_dir model/ --model_dir model/ --num_epoch 10
