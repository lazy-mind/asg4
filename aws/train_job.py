import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
import preprocess

## @params: [JOB_NAME]
args = getResolvedOptions(sys.argv, ['JOB_NAME'])

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)
## @type: DataSource
## @args: [database = "asg4_train", table_name = "train", transformation_ctx = "datasource0"]
## @return: datasource0
## @inputs: []
datasource0 = glueContext.create_dynamic_frame.from_catalog(database = "asg4_train", table_name = "train", transformation_ctx = "datasource0")
## @type: ApplyMapping
## @args: [mapping = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("date", "string", "date", "string"), ("querytype", "string", "querytype", "string"), ("userid", "string", "userid", "string"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1"]
## @return: applymapping1
## @inputs: [frame = datasource0]
applymapping1 = ApplyMapping.apply(frame = datasource0, mappings = [("sentiment", "long", "sentiment", "long"), ("twitterid", "long", "twitterid", "long"), ("date", "string", "date", "string"), ("querytype", "string", "querytype", "string"), ("userid", "string", "userid", "string"), ("tweet", "string", "tweet", "string")], transformation_ctx = "applymapping1")

## @type: Map
## @args: [f = map_function, transformation_ctx = "mapping1"]
## @return: mapping1
## @inputs: [frame = applymapping1]
def map_function(dynamicRecord):
    tweet = dynamicRecord["tweet"]
    features = preprocess.preprocess_text(tweet, 140)
    dynamicRecord["features"] = features
    return dynamicRecord
mapping1 = Map.apply(frame = applymapping1, f = map_function, transformation_ctx = "mapping1")

## @type: DropFields
## @args: [paths = ["date", "tweet", "querytype", "twitterid", "userid"], transformation_ctx = "mapping2"]
## @return: mapping2
## @inputs: [frame = mapping1]
mapping2 = DropFields.apply(frame = mapping1, paths = ["date", "tweet", "querytype", "twitterid", "userid"], transformation_ctx = "mapping2")


## @type: DataSink
## @args: [connection_type = "s3", connection_options = {"path": "s3://asg4/train_features"}, format = "json", transformation_ctx = "datasink2"]
## @return: datasink2
## @inputs: [frame = mapping2]
datasink2 = glueContext.write_dynamic_frame.from_options(frame = mapping2, connection_type = "s3", connection_options = {"path": "s3://asg4/train_features"}, format = "json", transformation_ctx = "datasink2")
job.commit()