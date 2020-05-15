#!/bin/bash

helpFunction()
{
   echo ""
   echo "Usage: $0 -b base_model"
   echo -e "\t-b Base model to re-train"
   exit 1 # Exit script after printing help
}

while getopts "b:" opt
do
   case "$opt" in
      b ) base_model="$OPTARG" ;;
      ? ) helpFunction ;; # Print helpFunction in case parameter is non-existent
   esac
done

# Print helpFunction in case parameters are empty
if [ -z "$base_model" ] 
then
   echo "Some or all of the parameters are empty";
   helpFunction
fi


#Resize images
python transform_image_resolution.py -d data/images/ -s 800 600
#Convert XML to CSV
python3 xml_to_csv.py -i data/images/train -o data/annotations/train_labels.csv
python3 xml_to_csv.py -i data/images/test -o data/annotations/test_labels.csv
#Generate TFRecords
python3 generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --image_dir=data/images  --output_path=data/annotations/train.record
python3 generate_tfrecord.py --csv_input=data/annotations/test_labels.csv --image_dir=data/images  --output_path=data/annotations/test.record
#Run preprocess step
mlflow run preprocess/ -b local --no-conda -e preprocess -P base-model="$base_model"
#Run train step
mlflow run training/ -b local --no-conda -e train config_file='ssd_mobilenet_v2_coco.config' num_train_steps=500 num_eval_steps=500
#Run conversion step
#TODO
