
import os
import json
import pandas as pd
from google.cloud import storage
import argparse

def read_from_gcs(bucket, filename, path):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.download_to_filename(os.path.join(path, filename))
    print("Data downloaded to ", os.path.join(path, filename))

def write_to_gcs(bucket, filename, path):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket)
    blob = bucket.blob(filename)
    blob.upload_from_filename(os.path.join(path, filename))
    print(path+filename, " Model written to ", bucket, " bucket.")

def download_bucket(bucket, folder):
  storage_client = storage.Client()
  for image_file in storage_client.list_blobs(bucket, prefix='images'):
    if image_file.name == 'images/':
      continue
    destination_uri = os.path.join(folder, image_file.name[7:]) 
    image_file.download_to_filename(destination_uri)

def main():
    # Defining and parsing the command-line arguments
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--input-bucket', type=str, help='Input bucket.', default='ai-vqc') 
    parser.add_argument('--output-bucket', type=str, help='output bucket', default='ai-vqc') 
    args = parser.parse_args()

    final_columns=['nose_x', 'right_eye_x', 'left_knee_x', 'right_elbow_x', 'left_eye_x', 'right_wrist_x', 'left_ear_x', 'left_elbow_x', 'left_ankle_x', 'right_ankle_x', 'left_hip_x', 'right_hip_x', 'left_shoulder_x', 'right_ear_x', 'left_wrist_x', 'right_knee_x', 'right_shoulder_x', 'nose_y', 'right_eye_y', 'left_knee_y', 'right_elbow_y', 'left_eye_y', 'right_wrist_y', 'left_ear_y', 'left_elbow_y', 'left_ankle_y', 'right_ankle_y', 'left_hip_y', 'right_hip_y', 'left_shoulder_y', 'right_ear_y', 'left_wrist_y', 'right_knee_y', 'right_shoulder_y', 'pose']
    final_df=[]
    final_df=pd.DataFrame(columns=final_columns) 

    download_bucket(args.input_bucket, '/src/jsons')

    for filename in os.listdir('/src/jsons/'):
      with open(os.path.join('/src/jsons/', filename)) as json_file:
        x = json.load(json_file)

      for dataframe in x['sample']:
        df=pd.DataFrame(dataframe['poses'])
        row_x=df.loc[[0]]
        row_y=df.loc[[1]]

        columns_x=[]
        for column in row_x:
          column_x=column.replace(" ", "_")+'_x'
          columns_x.append(column_x)
        row_x.columns=columns_x

        columns_y=[]
        for column in row_y:
          column_y=column.replace(" ", "_")+'_y'
          columns_y.append(column_y)
        row_y.columns=columns_y
        row_y=row_y.reset_index(drop=True)

        sample=pd.concat([row_x, row_y], axis=1)
        sample['pose']=dataframe['pose']

        final_df=final_df.append(sample)
        sample.head()
    
    final_df.to_csv('/src/final_df.csv')
    write_to_gcs(args.output_bucket, 'final_df.csv', '/src')

if __name__ == "__main__":
    main()
        