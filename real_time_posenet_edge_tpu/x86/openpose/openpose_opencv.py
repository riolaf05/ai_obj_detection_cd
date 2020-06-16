import cv2

threshold=0.7

# Specify the paths for the 2 files
protoFile = r"pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = r"pose_iter_160000.caffemodel"

# Read the network into Memory
net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

# Read image
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    ret , frame = cap.read()
    if(ret):
        print("Connesso alla webcam")
    else:
        print("Webcam non disponibile o ce n'è più di una")
        exit(0)
    
    # Specify the input image dimensions
    inWidth = 368
    inHeight = 368

    # Prepare the frame to be fed to the network
    inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight), (0, 0, 0), swapRB=False, crop=False)

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    # Set the prepared object as the input blob of the network
    net.setInput(inpBlob)

    output = net.forward()

    #The output is a 4D matrix :
    # *The first dimension being the image ID ( in case you pass more than one image to the network ).
    # *The second dimension indicates the index of a keypoint. The model produces Confidence Maps and Part Affinity maps which are all concatenated. For COCO model it consists of 57 parts – 18 keypoint confidence Maps + 1 background + 19*2 Part Affinity Maps. Similarly, for MPI, it produces 44 points. We will be using only the first few points which correspond to Keypoints.
    # *The third dimension is the height of the output map.
    # *The fourth dimension is the width of the output map.

    H = output.shape[2]
    W = output.shape[3]
    # Empty list to store the detected keypoints
    points = []
    for i in range(len()):
        # confidence map of corresponding body's part.
        probMap = output[0, i, :, :]

        # Find global maxima of the probMap.
        minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

        # Scale the point to fit on the original image
        x = (frameWidth * point[0]) / W
        y = (frameHeight * point[1]) / H

        if prob > threshold:
            cv2.circle(frame, (int(x), int(y)), 15, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 255), 3, lineType=cv2.LINE_AA)

            # Add the point to the list if the probability is greater than the threshold
            points.append((int(x), int(y)))
        else :
            points.append(None)

    cv2.imshow("Output-Keypoints",frame)
    cv2.waitKey(0)

cap.release()
cv2.destroyAllWindows()