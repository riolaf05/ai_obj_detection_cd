#In docker run command map <host_folder>/label/train:data/label/train and <host_folder>/label/test:data/label/test

# Convert train folder annotation xml files to a single csv file,
# generate the `label_map.pbtxt` file to `data/annotations` directory as well.
python3 xml_to_csv.py -i label/train -o annotations/train_labels.csv -l annotations

# Convert test folder annotation xml files to a single csv.
python3 xml_to_csv.py -i test/test -o annotations/test_labels.csv 

# Generate `train.record`
python3 generate_tfrecord.py --csv_input=annotations/train_labels.csv --output_path=annotations/train.record --img_path=resized_images/train --label_map annotations/label_map.pbtxt

# Generate `test.record`
python3 generate_tfrecord.py --csv_input=annotations/test_labels.csv --output_path=annotations/test.record --img_path=resized_images/test --label_map annotations/label_map.pbtxt
