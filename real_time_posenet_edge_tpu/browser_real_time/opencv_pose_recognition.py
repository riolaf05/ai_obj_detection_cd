#import tensorflow as tf
from flask import Flask, render_template, Response
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

#lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)

def pose_detect():
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
            engine = PoseEngine('/home/scripts/pose_detection/models/browser/posenet_mobilenet_v1_075_481_641_quant_decoder_edgetpu.tflite')
            poses, inference_time = engine.DetectPosesInImage(np.uint8(pil_image))
            print('Inference time: %.fms' % inference_time)

            '''
            for pose in poses:
                if pose.score < 0.4: continue
                print('\nPose Score: ', pose.score)
                for label, keypoint in pose.keypoints.items():
                    print(' %-20s x=%-4d y=%-4d score=%.1f' %
                        (label, keypoint.yx[1], keypoint.yx[0], keypoint.score))
            '''
            encode_return_code, image_buffer = cv2.imencode('.jpg', frame) #TODO: draw lines on body parts!
            io_buf = io.BytesIO(image_buffer)
            yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + io_buf.read() + b'\r\n')
            #cv2.imshow("video", frame)
            #if cv2.waitKey(1) & 0xFF == ord('q'):
            #    break

@app.route('/pose_detection')
def pose_detection():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(
        pose_detect(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, threaded=True)