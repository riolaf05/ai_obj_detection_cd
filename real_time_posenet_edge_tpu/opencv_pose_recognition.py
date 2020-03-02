#import tensorflow as tf
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

def main():
    #with tf.Session() as sess:
        #model_cfg, model_outputs = posenet.load_model(args.model, sess)
        #output_stride = model_cfg['output_stride']
        
        inWidth = 368
        inHeight = 368
        cam_width = 1280
        cam_height = 720
        cam_id = 0
        video_file = None
        threshold = 0.1
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        if video_file is not None:
            cap = cv2.VideoCapture(video_file)
        else:
            cap = cv2.VideoCapture(cam_id)
        cap.set(3, cam_width)
        cap.set(4, cam_height)
        success,frame = cap.read()
        count = 0
        while success:
            #cv2.imwrite(r"C:\Users\lafacero\Desktop\opencv_test\images\frame%d.jpg" % count, image)     # save frame as JPEG file
            success,frame = cap.read()
            print ('Read frame_{}'.format(count), " ", success)
            count += 1

            
            frame = cv2.cvtColor(cv2_im,cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame)
            pil_image.resize((641, 481), Image.NEAREST)
            engine = PoseEngine('models/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
            poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
            print('Inference time: %.fms' % inference_time)

            for pose in poses:
                if pose.score < 0.4: continue
                print('\nPose Score: ', pose.score)
                for label, keypoint in pose.keypoints.items():
                    print(' %-20s x=%-4d y=%-4d score=%.1f' %
                        (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))

            #cv2.imshow("video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()