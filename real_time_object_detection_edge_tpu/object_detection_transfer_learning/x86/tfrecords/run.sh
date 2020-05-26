#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -b base_model -c base_model_config -nt num_train_steps -ne num_eval_steps -d img_size"
   echo -e "\t-b Base model to re-train [ssd_mobilenet_v2, faster_rcnn_inception_v2, rfcn_resnet101]"
   echo -e "\t-c Base model configuration file [ssd_mobilenet_v2_coco.config, faster_rcnn_inception_v2_pets.config, rfcn_resnet101_pets.config]"
   echo -e "\t-nt Number of training epochs"
   echo -e "\t-ne Number of evaluation steps"
   echo -e "\t-n Number of test samples"
   echo -e "\t-d Image dimension"
   exit 1 # Exit script after printing help
}

while getopts "b:c:nt:ne:" opt
do
   case "$opt" in
      b ) base_model="$OPTARG" ;;
      c ) base_model_config="$OPTARG" ;;
      nt ) num_train_steps="$OPTARG" ;;
      ne ) num_eval_steps="$OPTARG" ;;
      n ) num_example="$OPTARG" ;;
      d ) img_size="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$base_model" ] || [ -z "$base_model_config" ] || [ -z "$num_train_steps" ] || [ -z "$num_eval_steps" ] || [ -z "$img_size" ] || [ -z "$num_example" ] 
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi

#Resize images
python transform_image_resolution.py -d data/images/train -s 800 600
python transform_image_resolution.py -d data/images/test -s 800 600
#Convert XML to CSV (train and test data)
python3 xml_to_csv.py -i data/images/train -o data/annotations/train_labels.csv
python3 xml_to_csv.py -i data/images/test -o data/annotations/test_labels.csv
#Generate TFRecords (train and test data)
python3 generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --image_dir=data/images/train  --output_path=data/annotations/train.record
python3 generate_tfrecord.py --csv_input=data/annotations/test_labels.csv --image_dir=data/images/test  --output_path=data/annotations/test.record
#Run preprocess step
mlflow run preprocess/ -b local --no-conda -e preprocess -P base-model=$base_model -P num-example=$num_example
#Run train step
mlflow run training/ -b local --no-conda -e train -P config_file=$base_model_config -P num_train_steps=$num_train_steps -P num_eval_steps=$num_eval_steps
#Run conversion step
mlflow run training/ -b local --no-conda -e convert -P config_file=$base_model_config -P img_size=$img_size 
