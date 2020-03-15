# Convert train folder annotation xml files to a single csv file,
# generate the `label_map.pbtxt` file to `data/annotations` directory as well.
python3 xml_to_csv.py -i data/label/train -o data/annotations/train_labels.csv -l data/annotations

# Convert test folder annotation xml files to a single csv.
python3 xml_to_csv.py -i data/test/test -o data/annotations/test_labels.csv -l data/annotations

# Generate `train.record`
python3 generate_tfrecord.py --csv_input=data/annotations/train_labels.csv --output_path=data/annotations/train.record --img_path=data/resized_images/train --label_map data/annotations/label_map.pbtxt

# Generate `test.record`
python3 generate_tfrecord.py --csv_input=data/annotations/test_labels.csv --output_path=data/annotations/test.record --img_path=data/resized_images/test --label_map data/annotations/label_map.pbtxt
