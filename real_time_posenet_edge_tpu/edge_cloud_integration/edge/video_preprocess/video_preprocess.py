import cv2
import os
import argparse

def frame_extraction(cap, video_name):
  i=0
  while(cap.isOpened()):
      ret, frame = cap.read()
      if ret == False:
          break
      cv2.imwrite('/images/'+video_name+'_frame'+str(i)+'.jpg',frame)
      i+=1

def main():
    parser = argparse.ArgumentParser(description='Train arguments')
    parser.add_argument('--video', type=str, help='Video to preprocess', default=False)
    args = parser.parse_args()

    cap= cv2.VideoCapture(os.path.join('video', args.video))
    frame_extraction(cap, args.video)

if __name__ == "__main__":
    main()

