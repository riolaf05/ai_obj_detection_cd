import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import argparse

BASE_DIR='/home/scripts/pose_detection'

def main():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--save-gdatastore', type=bool, help='True to save on Google Datastore', default=False) 
    args = parser.parse_args()

    images = os.listdir(os.path.join(BASE_DIR, 'images'))
    for image in images:
        pil_image = Image.open(os.path.join(BASE_DIR, 'images', image))
        pil_image.resize((641, 481), Image.NEAREST)
        engine = PoseEngine(os.path.join(BASE_DIR,'models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite'))
        poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
        print('Inference time: %.fms' % inference_time)

        for pose in poses:
            if pose.score < 0.4: continue
            print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                print(' %-20s x=%-4d y=%-4d score=%.1f' %
                    (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
                    #if arg.save_gdatastore==True:
                        #
                        
if __name__ == "__main__":
    main()

