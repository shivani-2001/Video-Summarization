# import packages
import cv2
import time
import numpy as np

video = cv2.VideoCapture('1.mp4') # capture first video
video1 = cv2.VideoCapture('2.mp4') # capture second video

threshold = 5. # set threshold value to compare frames

writer = cv2.VideoWriter('output1.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, (512, 512)) # to create a new video

ret, frame1 = video.read() # read first video
current_frame1 = frame1 # take first frame from video
current_frame1 = cv2.resize(current_frame1, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_CUBIC) # resize the size of frame

ret, frame2 = video1.read() # read second video
current_frame2 = frame2 # take first frame from video
current_frame2 = cv2.resize(current_frame2, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_CUBIC) # resize the size of frame

# Built-in weights and model for human detection
protopath = 'MobileNetSSD_deploy.prototxt'
modelpath = 'MobileNetSSD_deploy.caffemodel'

detector = cv2.dnn.readNetFromCaffe(prototxt=protopath, caffeModel=modelpath) # define detector

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

start_time1 = time.time()
while True:
    """
    It will read two videos parallelly and then for every frame firstly it will detect humans, 
    if humans are in frame the it will compare frame to previous frame, then compare images and store in new video
    
    return - summarized video  
    """

    # create new frames
    frame1_add = 255*np.ones(shape=[512, 512, 3], dtype=np.uint8)
    frame2_add = 255*np.ones(shape=[512, 512, 3], dtype=np.uint8)
    frame3_add = 255 * np.ones(shape=[512, 512, 3], dtype=np.uint8)

    ret1, frame_1 = video.read() # read first video
    ret2, frame_2 = video1.read() # read second video

    if ret1 is True:
        frame_1 = cv2.resize(frame_1, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_CUBIC) # resize the size of frame
        (H, W) = frame_1.shape[:2] # height and width of frame

        # start_time2 = time.time()
        if (((np.sum(np.absolute(frame_1 - current_frame1)) / np.size(frame_1)) > threshold)):  # compare frame to previous frame
            # end_time2 = time.time()
            # print("Frame comparison Time1: ", end_time2-start_time2)

            # human detection in every frame
            blob = cv2.dnn.blobFromImage(frame_1, 0.007843, (W, H), 127.5)

            detector.setInput(blob)
            person_detections = detector.forward()

            # start_time3 = time.time()
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections[0, 0, i, 1])

                    if CLASSES[idx] == "person":
                        # person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        # (startX, startY, endX, endY) = person_box.astype("int")
                        frame1_add = frame_1
            # end_time3 = time.time()
            # print("Human detection Time1: ", end_time3 - start_time3)
        else:
            current_frame1 = frame_1

        # cv2.imshow('frame1', frame_1)

    if ret2 is True:
        frame_2 = cv2.resize(frame_2, (512, 512), fx=0, fy=0, interpolation = cv2.INTER_CUBIC) # resize the size of frame
        (H, W) = frame_2.shape[:2] # height and width of frame

        # start_time4 = time.time()
        if (((np.sum(np.absolute(frame_2 - current_frame2)) / np.size(frame_2)) > threshold)):  # compare frame to previous frame
            # end_time4 = time.time()
            # print("Frame comparison Time2: ", end_time4 - start_time4)

            # human detection in every frame
            blob = cv2.dnn.blobFromImage(frame_2, 0.007843, (W, H), 127.5)

            detector.setInput(blob)
            person_detections = detector.forward()

            # start_time5 = time.time()
            for i in np.arange(0, person_detections.shape[2]):
                confidence = person_detections[0, 0, i, 2]
                if confidence > 0.5:
                    idx = int(person_detections[0, 0, i, 1])

                    if CLASSES[idx] == "person":
                        # person_box = person_detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        # (startX, startY, endX, endY) = person_box.astype("int")
                        frame2_add = frame_2
            # end_time5 = time.time()
            # print("Human detection Time2: ", end_time3 - start_time3)
        else:
            current_frame2 = frame_2

        cv2.imshow('frame2', frame_2)

    # if we get both frames then merge them and then store.
    if (((np.sum(np.absolute(frame1_add - frame3_add)) / np.size(frame1_add)) and
        (np.sum(np.absolute(frame2_add - frame3_add)) / np.size(frame2_add))) > threshold):

        final_frame = cv2.addWeighted(frame1_add, .5, frame2_add, .5, 0)
        writer.write(final_frame)

    # if we get first frame only then simply store first frame.
    elif ((np.sum(np.absolute(frame1_add - frame3_add)) / np.size(frame1_add)) > threshold):
        writer.write(frame1_add)

    # if we get second frame only then simply store second frame.
    elif ((np.sum(np.absolute(frame2_add - frame3_add)) / np.size(frame2_add)) > threshold):
        writer.write(frame2_add)

    if cv2.waitKey(1) & 0xFF == ord('q'): # to stop program press "Q" or "q"
        break

    if (np.shape(frame_1) == ()) and (np.shape(frame_2) == ()):
        break

end_time1 = time.time()
# release all the resources
video.release()
writer.release()
cv2.destroyAllWindows()
print("Total time: ", end_time1-start_time1)
