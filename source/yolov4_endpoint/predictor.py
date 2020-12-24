# -*- coding: utf-8 -*-
import sys
import json
import boto3
import os
import warnings
import flask
import cv2
import _thread
import time
import imutils
from imutils.video import FPS
import numpy as np

warnings.filterwarnings("ignore",category=FutureWarning)
import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass

confidence_baseline = 0.5
threshold_baseline = 0.3

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task


# The flask app for serving predictions
app = flask.Flask(__name__)
s3_client = boto3.client('s3')

weight = './yolov4.weights'
names = './coco.names'
cfg = './yolov4.cfg'

# load the COCO class labels our YOLO model was trained on

LABELS = open('./coco.names').read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
    dtype="uint8")

net = cv2.dnn.readNetFromDarknet(cfg, weight)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

outNames = net.getUnconnectedOutLayersNames()
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

layerNames = net.getLayerNames()
lastLayerId = net.getLayerId(layerNames[-1])
lastLayer = net.getLayer(lastLayerId)

# model = cv.dnn.readNet(weight,cfg)
# model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
# model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
# net = cv.dnn_DetectionModel(model)

# net.setInputSize(416, 416)
# net.setInputScale(1.0 / 255)
# net.setInputSwapRB(True)

# 图片检测
def yolo_infer(weight,cfg,pic):
    print ("<<<<start")
    #TODO: define endpoint output
    # 调用训练的输出weight
    #net = cv.dnn_DetectionModel(cfg,weight)

    frame = cv2.imread(pic)
    # for i in range(1,10):
    start_time = time.time()
    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    end_time = time.time()
    print("infer time " + str(end_time-start_time))
    #delete_file(pic)
    return classes,confidences,boxes




# 视频检测
def yolo_infer_for_video(weight, cfg, frame):
    #net = cv.dnn_DetectionModel(cfg,weight)

    # for performance reason, comments these lines by lch
    # model = cv.dnn.readNet(weight,cfg)
    # model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
    # net = cv.dnn_DetectionModel(model)

    # net.setInputSize(416, 416)
    # net.setInputScale(1.0 / 255)
    # net.setInputSwapRB(True)

    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    return classes,confidences,boxes


def detect_objects(bucket, weight,names,cfg,video,output_s3_prefix):
    # get video frames and pass to YOLO for output
    video_startime= time.time()
    vs = cv2.VideoCapture(video)
    writer = None
    (W, H) = (None, None)
    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int(vs.get(prop))
        print("[INFO]video file {} has {} total frames in video".format(video,total))

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
        
    frame_num=0
    # initialize video stream, pointer to output video file and grabbing frame dimension
    names_li = []
    with open(names) as file_obj:
        for line in file_obj.readlines():  
            names_li.append(line.strip())
            
    file_name_fre = ''.join(video.split('.')[:-1])
    # 生成文件名
    txt_file = file_name_fre+".txt"
    f_txt = open(txt_file, 'a')
    print ("tmp txt file is "+txt_file)
    
    # while(cap.isOpened()):
    while True:
        
        (ret, frame) = vs.read()
        frame_num = frame_num+1


        if not ret:
            break;
        if ret:
            # only process odd frame, it will speed up and reduce video file size
            # if frame_num % 2 ==0:
            #     continue;
            # if frame_num == 1501:
            #     break;
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),swapRB=True, crop=False)
            net.setInput(blob)         
            layerOutputs = net.forward(ln)

            # initialize our lists of detected bounding boxes, confidences,
            # and class IDs, respectively
            boxes = []
            confidences = []
            classIDs = []

            # loop over each of the layer outputs
            for output in layerOutputs:
                # loop over each of the detections
                for detection in output:
                    # extract the class ID and confidence (i.e., probability)
                    # of the current object detection
                    scores = detection[5:]
                    classID = np.argmax(scores)
                    confidence = scores[classID]

                    # filter out weak predictions by ensuring the detected
                    # probability is greater than the minimum probability
                    if confidence > confidence_baseline:
                        # scale the bounding box coordinates back relative to
                        # the size of the image, keeping in mind that YOLO
                        # actually returns the center (x, y)-coordinates of
                        # the bounding box followed by the boxes' width and
                        # height
                        box = detection[0:4] * np.array([W, H, W, H])
                        (centerX, centerY, width, height) = box.astype("int")

                        # use the center (x, y)-coordinates to derive the top
                        # and and left corner of the bounding box
                        x = int(centerX - (width / 2))
                        y = int(centerY - (height / 2))

                        # update our list of bounding box coordinates,
                        # confidences, and class IDs
                        boxes.append([x, y, int(width), int(height)])
                        confidences.append(float(confidence))
                        classIDs.append(classID)

            # apply non-maxima suppression to suppress weak, overlapping
            # bounding boxes
            idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_baseline, threshold_baseline)

            # ensure at least one detection exists
            if len(idxs) > 0:
                # loop over the indexes we are keeping
                object_detection_desc = ''
                for i in idxs.flatten():
                    # extract the bounding box coordinates
                    (x, y) = (boxes[i][0], boxes[i][1])
                    (w, h) = (boxes[i][2], boxes[i][3])

                    # draw a bounding box rectangle and label on the frame
                    color = [int(c) for c in COLORS[classIDs[i]]]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    label = "{}: {:.4f}".format(LABELS[classIDs[i]],confidences[i])
                    cv2.putText(frame, label, (x, y - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    #writer.write(frame)       
                    object_detection_desc = object_detection_desc + '%s left_x: %.4f top_y: %.4f width: %.4f height: %.4f \n' %(label,x,y,w,h)             
                    #object_detection_desc = str(str(label) +' (left_x:'+ str(x)+' top_y:'+str(y)+ ' width:'+ str(w)+' height:'+str(h)+')')

#                 print(object_detection_desc)
                f_txt.write(object_detection_desc)    
                    
            
            # check if the video writer is None
            if writer is None:
                # initialize our video writer
                fourcc = cv2.VideoWriter_fourcc('x','2','6','4')
                print ("tmp video file ="+file_name_fre+'_res.mp4')
                writer = cv2.VideoWriter(file_name_fre+'_res.mp4', fourcc, 30,
                    (frame.shape[1], frame.shape[0]), True)
#                 writer = cv2.VideoWriter(file_name_fre+'_res.mp4', 0x21, 30,
#                     (frame.shape[1], frame.shape[0]), True)

                # some information on processing single frame
                if total > 0:
                    #elap = (end - start)
                    print("[INFO] single frame took  seconds")
                    #print("[INFO] estimated total time to finish: {:.4f}".format(elap * total))
            writer.write(frame) 
        # frame_endtime=time.time()
        # print('Frame process time %.4f' %(frame_endtime-frame_startime))

    
    video_endtime = time.time()
    print ("<<<<video {} infer done, use time {} ".format(video,str(video_endtime-video_startime)))
    # 必须关闭txt文件
    f_txt.close()
    writer.release()
    vs.release()
    
    try:
        print ('file_name_fre='+file_name_fre)
        print ('txt_file='+txt_file)
        s3_client.upload_file(file_name_fre+'_res.mp4', bucket, output_s3_prefix+'/'+file_name_fre.split('/')[-1:][0].split('.')[0]+'_res.mp4')
        print('s3 upload video file' +  file_name_fre+'_res.mp4' + "--" +bucket + "--" + output_s3_prefix+'/'+file_name_fre.split('/')[-1:][0].split('.')[0]+'_res.mp4')
        s3_client.upload_file(txt_file, bucket, output_s3_prefix+'/'+txt_file.split('/')[-1:][0])
        print('s3 upload text file' + txt_file + "--" +bucket + "--" + output_s3_prefix+ '/'+txt_file.split('/')[-1:][0])
        delete_file(video)
        delete_file(file_name_fre+'_res.mp4')
        delete_file(txt_file)
        print('clean file'+video + ', '+file_name_fre+'_res.mp4' +","+txt_file)
    except Exception as e:
#         print(e)
        print('done')





def delete_file(file):
    """
    delete file
    :param file:
    :return:
    """
    if os.path.isfile(file):
        try:
            os.remove(file)
        except:
            pass   
        
        
# def yolo_infer(weight,names,cfg,pic):
#     #TODO: define endpoint output
#     frame = cv.imread(pic)
#     #print ("<<<<pic shape:", frame.shape)
#     #print(weight,cfg)
#     model = cv.dnn.readNet(weight,cfg)
#     model.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#     model.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA_FP16)
#     print(model)
#     net = cv.dnn_DetectionModel(model)
#     #print(net)
#     net.setInputSize(608, 608)
#     net.setInputScale(1.0 / 255)
#     net.setInputSwapRB(True)
#     with open(names, 'rt') as f:
#         names = f.read().rstrip('\n').split('\n')

#     classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
#     #print(classes)
#     #print(confidences)
#     #print(boxes)
#     print ("<<<<done")
#     return classes,confidences,boxes


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    # health = ScoringService.get_model() is not None  # You can insert a health check here
    health = 1

    status = 200 if health else 404
    # print("===================== PING ===================")
    return flask.Response(response="{'status': 'Healthy'}\n", status=status, mimetype='application/json')

@app.route('/invocations', methods=['POST'])
def invocations():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    data = None
    print("================ INVOCATIONS =================")

    #parse json in request
    print ("<<<< flask.request.content_type", flask.request.content_type)

    data = flask.request.data.decode('utf-8')
    data = json.loads(data)
    print(data)

    bucket = data['bucket']
    s3_url = data['s3_url']
    # 不论是图片/视频 都必须下载 因为这里都涉及到读取视频/图片【最后也必须删除这玩意】
    try:
        input_file_name = s3_url.split('/')[-1]
        download_tmp_file_name='/opt/ml/'+input_file_name
        s3_client.download_file(bucket, s3_url, download_tmp_file_name)
    except:
        #local test
        download_tmp_file_name= '/opt/program/'+data['s3_url']
    
#     print('Download finished!')
    # inference and send result to RDS and SQS
    print('Start to inference:')

    #LOAD MODEL
    weight = './yolov4.weights'
    names = './coco.names'
    cfg = './yolov4.cfg'
    

    #make sure the model parameters exist
    for i in [weight,names,cfg]:
        if os.path.exists(i):
            print ("<<<<pretrained model exists for :", i)
        else:
            print ("<<< make sure the model parameters exist for: ", i)
            break
    # 图片推理 make inference
    if data['type'] == 'pic':
        print('infer pic')
        classes, confidences, boxes = yolo_infer(weight, cfg, input_file_name)
        print ("Done inference picture! ")
        inference_result = {
            'classes':classes.tolist(),
            'confidences':confidences.tolist(),
            'boxes':boxes.tolist()
        }
        _payload = json.dumps(inference_result,ensure_ascii=False)
    else:
        output_s3_prefix = data['output_s3_prefix']
        print('infer video')
        _thread.TIMEOUT_MAX = 9999999
#         detect_objects(bucket, weight, names, cfg, download_file_name,output_s3_prefix)
        _thread.start_new_thread(detect_objects, (bucket, weight, names, cfg, download_tmp_file_name,output_s3_prefix))
        print ("Done inference video! ")
        inference_result = {
            'vidoe':'infer start!——>please go to cloudformation to see the process is done or not!!'
        }
        _payload = json.dumps(inference_result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')

