#import tensorflow as tf
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

MODE = "MPI"

if MODE is "COCO":
    protoFile = r"C:\Users\lafacero\Desktop\opencv_test\model\pose_deploy_linevec.prototxt"
    weightsFile = r"C:\Users\lafacero\Desktop\opencv_test\model\pose_iter_440000.caffemodel"
    nPoints = 18
    POSE_PAIRS = [ [1,0],[1,2],[1,5],[2,3],[3,4],[5,6],[6,7],[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],[0,14],[0,15],[14,16],[15,17]]

elif MODE is "MPI" :
    protoFile = r"C:\Users\lafacero\Desktop\opencv_test\model\pose_deploy_linevec_faster_4_stages.prototxt"
    weightsFile = r"C:\Users\lafacero\Desktop\opencv_test\model\pose_iter_160000.caffemodel"
    nPoints = 15
    POSE_PAIRS = [[0,1], [1,2], [2,3], [3,4], [1,5], [5,6], [6,7], [1,14], [14,8], [8,9], [9,10], [14,11], [11,12], [12,13] ]

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

            # Next, we find the keypoints for a image with only single person
            #frame= cv2.imread(r"C:\Users\lafacero\Desktop\opencv_test\images\flessioni.jpg")
            frameCopy = np.copy(frame)
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            threshold = 0.1

            # Pass it through the network
            inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

            net.setInput(inpBlob)

            output = net.forward()
            H = output.shape[2]
            W = output.shape[3]

            # gather the points and plot the keypoints and the skeleton figure¶
            # Empty list to store the detected keypoints
            points = []

            for i in range(nPoints):
                # confidence map of corresponding body's part.
                probMap = output[0, i, :, :]

                # Find global maxima of the probMap.
                minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)
                
                # Scale the point to fit on the original image
                x = (frameWidth * point[0]) / W
                y = (frameHeight * point[1]) / H

                if prob > threshold : 
                    cv2.circle(frameCopy, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                    cv2.putText(frameCopy, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, lineType=cv2.LINE_AA)
                    cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                    # Add the point to the list if the probability is greater than the threshold
                    points.append((int(x), int(y)))
                else :
                    points.append(None)

            # Draw Skeleton
            for pair in POSE_PAIRS:
                partA = pair[0]
                partB = pair[1]

                if points[partA] and points[partB]:
                    cv2.line(frame, points[partA], points[partB], (0, 255, 255), 3)


            cv2.imshow(MODE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    main()