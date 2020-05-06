import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import argparse
from google.cloud import storage
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.2f')

BASE_DIR='/home/scripts/pose_detection'

def write_to_gcs(filename, path, bucket_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(filename)
    blob.upload_from_filename(path+filename)
    print(path+filename, " written to ", bucket_name, " bucket.")

def main():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--bucket', type=str, help='Bucket to save on Google Datastore', default=False)
    parser.add_argument('--pose', type=str, help='Pose in the video', default=False)
    args = parser.parse_args()

    engine = PoseEngine(os.path.join(BASE_DIR,'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'))

    #Processing frames
    images = os.listdir(os.path.join(BASE_DIR, 'images'))
    for image in images:
        pil_image = Image.open(os.path.join(BASE_DIR, 'images', image))
        pil_image.resize((641, 481), Image.NEAREST)
        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image)) 
        print('Inference time: %.fms' % inference_time)

        data = {}
        data['sample'] = []
        poses_list={}
        for pose in poses: #TODO: gestire presenza multipla nelle immagini
            if pose.score < 0.4: continue
            #print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                poses_list[label] = []
                poses_list[label].append(keypoint.yx[1])
                poses_list[label].append(keypoint.yx[0])
                poses_list[label].append(keypoint.score)
                #print(' %-20s x=%-4d y=%-4d score=%.1f' %
                    #(label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
                    #if arg.save_gdatastore==True:
                        #
            data['sample'].append({
                'score': pose.score,
                'poses': poses_list,
                'pose' : args.pose
            })
        print(data)
        with open('outputs/{}.txt'.format(image[:-4]), 'w') as f:
            json.dump(data, f)
        if args.bucket:
            write_to_gcs('outputs/{}.txt'.format(image[:-4]), '/home/scripts/pose_detection/images', args.bucket)

if __name__ == "__main__":
    main()

