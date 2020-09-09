import os
import numpy as np
from PIL import Image
from pose_engine import PoseEngine
import argparse
from scipy.spatial import distance as dist
import cv2

BASE_DIR='/home/scripts/pose_detection'

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

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
        
        subjects=[]
        for pose in poses:
            dict={}
            if pose.score < 0.4: continue
            print('\nPose Score: ', pose.score)
            for label, keypoint in pose.keypoints.items():
                print(' %-20s x=%-4d y=%-4d score=%.1f' % (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
                dict[label]=[keypoint.yx[1], keypoint.yx[0]]
                subjects.append(dict)
                    #if arg.save_gdatastore==True:
                        #

        image = cv2.imread(os.path.join(BASE_DIR, 'images', image))
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        radius = 2
        color = (255, 0, 0) 
        thickness = 1

        for label in subjects[0].items():
            coord = [int(i) for i in label[1]]
            image = cv2.circle(image, tuple(coord), radius, color, thickness)
            #cv2.putText(image,label[0],tuple(label[1]), font, 0.5,(255,255,255),1,cv2.LINE_AA)
        
        #Link points for the first subject
        cv2.line(image,tuple(subjects[0]['left hip']),tuple(subjects[0]['left knee']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['left knee']),tuple(subjects[0]['left ankle']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['right hip']),tuple(subjects[0]['right knee']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['right knee']),tuple(subjects[0]['right ankle']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['left hip']),tuple(subjects[0]['right hip']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['left shoulder']),tuple(subjects[0]['right shoulder']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['left shoulder']),tuple(subjects[0]['left elbow']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['left elbow']),tuple(subjects[0]['left wrist']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['right shoulder']),tuple(subjects[0]['right elbow']),(255,0,0),5)
        cv2.line(image,tuple(subjects[0]['right elbow']),tuple(subjects[0]['right wrist']),(255,0,0),5)

        #Distance between points
        # compute the Euclidean distance between the coordinates,
        # and then convert the distance in pixels to distance in
        # units
        D = dist.euclidean(tuple(subjects[0]['left knee']), tuple(subjects[0]['left knee']))
        (mX, mY) = midpoint((xA, yA), (xB, yB))
        cv2.putText(orig, "{:.1f}in".format(D), (int(mX), int(mY - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        print("Distance: ", D)

        cv2.imwrite(os.path.join(BASE_DIR, 'images', image+'_inf.jpg'), image) 

if __name__ == "__main__":
    main()