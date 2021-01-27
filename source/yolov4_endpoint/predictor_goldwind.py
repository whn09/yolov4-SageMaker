# -*- coding: utf-8 -*-
import sys
import json
import boto3
import os
import warnings
warnings.filterwarnings("ignore",category=FutureWarning)
import _thread

# sys.path.append('/opt/program/textrank4zh')

import sys
try:
    reload(sys)
    sys.setdefaultencoding('utf-8')
except:
    pass


with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    # from autogluon import ImageClassification as task

import flask
import cv2 as cv

# The flask app for serving predictions
app = flask.Flask(__name__)

s3_client = boto3.client('s3')

import cv2 as cv

# 图片检测
def yolo_infer(bucket, weight,names,cfg,pic):
    print ("<<<<start")
    #TODO: define endpoint output
    # 调用训练的输出weight
    net = cv.dnn_DetectionModel(cfg,weight)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    frame = cv.imread(pic)
    with open(names, 'rt') as f:
        names = f.read().rstrip('\n').split('\n')
    # 下面的文件 只会写到镜像中
    f_img = open('./result_helmat.txt','w')
    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
#     for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
#         label = '%.2f' % confidence
#         label = '%s: %s' % (names[classId], label)
#         labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
#         left, top, width, height = box
#         top = max(top, labelSize[1])
#         cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
#         cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
#         cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))        
#         # 写入json文件
#         img_json = str(str(label) +' (left_x:'+ str(left)+' top_y:'+str(top)+ ' width:'+ str(width)+' height:'+str(height)+')')
#         print(img_json)
#         f_img.write(img_json)
#         f_img.write('\n')
#     f_img.close()
#     # 识别后的图片 保存成jpg 保存到镜像中
#     cv.imwrite('./result_helmat.jpg', frame)
#     s3_client.upload_file('./result_helmat.jpg', bucket, 'output/res.jpg')
#     print ("<<<<done")
    return classes,confidences,boxes

import time
import imutils


# 视频检测
def yolo_infer_for_video(weight,cfg,frame):
    net = cv.dnn_DetectionModel(cfg,weight)
    net.setInputSize(608, 608)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    classes, confidences, boxes = net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    return classes,confidences,boxes


def detect_objects(bucket, weight,names,cfg,video):
    # get video frames and pass to YOLO for output
    cap = cv.VideoCapture(video)
    writer = None
    # try to determine the total number of frames in the video file
    try:
        prop = cv.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv.CAP_PROP_FRAME_COUNT
        total = int(cap.get(prop))
        print("[INFO] {} total frames in video".format(total))

    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1
        
    i=0
    # initialize video stream, pointer to output video file and grabbing frame dimension
    names_li = []
    with open(names) as file_obj:
        for line in file_obj.readlines():  
            names_li.append(line.strip())
            
    while(cap.isOpened()):
        stime= time.time()
        ret, frame = cap.read()
        i = i+1
        print (i)
        if ret:
            classes, confidences, boxes = yolo_infer_for_video(weight,cfg,frame)
            end = time.time()
            if len(classes) == 0:
                writer.write(frame)
                continue
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                label = '%.2f' % confidence
                label = '%s: %s' % (names_li[classId], label)
                print('label——',label)
                labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                left, top, width, height = box
                print('box——',box)
                top = max(top, labelSize[1])
                cv.rectangle(frame, box, color=(0, 255, 0), thickness=3)
                cv.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv.FILLED)
                cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

            if writer is None:
                # Initialize the video writer
                fourcc = cv.VideoWriter_fourcc(*"MP4V")
                writer = cv.VideoWriter('./res.mp4', fourcc, 30,(frame.shape[1], frame.shape[0]), True)
                # some information on processing single frame
                if total > 0:
                    elap = (end - stime)
                    print("[INFO] single frame took {:.4f} seconds".format(elap))
                    print("[INFO] estimated total time to finish: {:.4f}".format(
                        elap * total))

            writer.write(frame)

            print('FPS {:1f}'.format(1/(time.time() -stime)))
            #  监听键盘，按下q键退出
            if cv.waitKey(1)  & 0xFF == ord('q'):
                break
        else:
            break
    s3_client.upload_file('./res.mp4', bucket, 'output/res.mp4')

    print ("<<<<video infer done")
    writer.release()
    cap.release()


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
    print("===================== PING ===================")
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
    download_file_name = s3_url.split('/')[-1]
    print ("<<<<download_file_name ", download_file_name)
#     s3_client.download_file(bucket, s3_url, download_file_name)
    
    #local test
    download_file_name= data['s3_url']
    
    print('Download finished!')
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
        classes, confidences, boxes = yolo_infer(bucket, weight, names, cfg, download_file_name)
        print ("Done inference picture! ")
        inference_result = {
            'classes':classes.tolist(),
            'confidences':confidences.tolist(),
            'boxes':boxes.tolist()
        }
        _payload = json.dumps(inference_result,ensure_ascii=False)
    else:
        print('infer video')
#         detect_objects(bucket, weight, names, cfg, download_file_name)
        output_s3_path = 'xxxxx'
        _thread.start_new_thread(detect_objects, (bucket, weight, names, cfg, download_file_name))
        print ("Done inference video! ")
        inference_result = {
            'vidoe':'infer is done!!',
            'output_s3_path':output_s3_path
        }
        _payload = json.dumps(inference_result,ensure_ascii=False)

    return flask.Response(response=_payload, status=200, mimetype='application/json')
