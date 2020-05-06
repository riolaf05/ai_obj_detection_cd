import sys
#See https://www.kubeflow.org/docs/pipelines/sdk/component-development/
import kfp
from kfp import compiler
import kfp.components as comp
import kfp.dsl as dsl
from kfp import gcp

EXPERIMENT_NAME = 'gpu-test'
BUCKET = "ai-vqc-kubeflow-components"
OUTPUT_BUCKET='ai-vqc-kubeflow-output'
PROJECT='ai-vqc'

#Component definition
def training_op(image_data_format,
               output,
               ):
  return dsl.ContainerOp(
    name='gpu-op', 
    image='eu.gcr.io/ai-vqc/x86-keras-gpu:latest',
    command="python3",
    arguments=[
        "/src/main.py",
        "--image-data-format", image_data_format,
        "--output", output,
    ],
    #pvolumes={"/mnt": download_step.pvolume}
    file_outputs={
        'accuracy-score': '/mlpipeline-metrics.json'
    }
  ).set_gpu_limit(1).apply(kfp.gcp.use_gcp_secret('user-gcp-sa'))


@dsl.pipeline(
  name='Kubeflow Test Pipeline',
  description='Performs preprocessing, training and deployment.'
)
def pipeline(
    image_data_format='channels_first',
    output='gs://ai-vqc-kubeflow-output/gpu/model',
    ):

    #Pipeline component instances
    train_gpu=training_op(image_data_format, output)

#Compile the pipeline
pipeline_func = pipeline('channels_first')
pipeline_filename = pipeline_func.__name__ + '.pipeline.zip'

import kfp.compiler as compiler
compiler.Compiler().compile(pipeline_func, pipeline_filename) #compiles your Python domain-specific language (DSL) code into a single static configuration (in YAML format) that the Kubeflow Pipelines service can process.

#Create Kubeflow experiment
client = kfp.Client()
try:
    experiment = client.get_experiment(experiment_name=EXPERIMENT_NAME)
except:
    experiment = client.create_experiment(EXPERIMENT_NAME)
    
print(experiment)

#Run the pipeline
arguments = {}
run_name = pipeline_func.__name__ + ' run'
run_result = client.run_pipeline(experiment.id, 
                                 run_name, 
                                 pipeline_filename, 
                                 arguments)
print(experiment.id)
print(run_name)
print(pipeline_filename)
print(arguments)
