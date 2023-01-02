import cv2
import numpy as np
import os
import bleedfacedetector as fd
import time
import csv

from automation import visualizationData
from flask import Flask, render_template, url_for, redirect, request
from datetime import datetime


app = Flask(__name__)

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# model = 'model/emotion-ferplus-8.onnx'
model = 'model/emotion-ferplus-8-final.onnx'


pathcsv = 'static/dataRecord/csv/'
visualPath = 'static/dataRecord/visualization/'

# Face Expression Detection

def init_emotion():
    
    # Set global variables
    global net,emotions
    
    # Define the emotions
    emotions = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Anger', 'Disgust', 'Fear', 'Contempt']
    
    # Initialize the DNN module
    net = cv2.dnn.readNetFromONNX(model)
    

def emotion(image, returndata=False, confidence=0.3):
    
    # Make copy of  image
    img_copy = image.copy()
    
    # Detect face in image
    faces = fd.ssd_detect(img_copy,conf=confidence)
    
    # Define padding for face ROI
    padding = 3 
    
    # Iterate process for all detected faces
    for x,y,w,h in faces:
        
        # Get the Face from image
        face = img_copy[y-padding:y+h+padding,x-padding:x+w+padding]
        
        # Convert the  detected face from BGR to Gray scale
        gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        
        # Resize the gray scale image into 64x64
        resized_face = cv2.resize(gray, (64, 64))
        
        # Reshape the final image in required format of model
        processed_face = resized_face.reshape(1,1,64,64)
        
        # Input the processed image
        net.setInput(processed_face)
        
        # Forwards pass
        Output = net.forward()
        # print(Output)
        
        #Compute softmax values for each sets of scores  
        expanded = np.exp(Output - np.max(Output))
        probablities =  expanded / expanded.sum()

        # Get the final probablities 
        prob = np.squeeze(probablities)
        
        # Get the predicted emotion
        global predicted_emotion
        predicted_emotion = emotions[prob.argmax()]        
        print(predicted_emotion)  
              
       
        # Write predicted emotion on image
        cv2.putText(img_copy,'{}'.format(predicted_emotion),(x,y+h+(1*20)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 
                        2, cv2.LINE_AA)
        # Draw rectangular box on detected face
        cv2.rectangle(img_copy,(x,y),(x+w,y+h),(0,0,255),2)
    
    if  returndata:
        # Return the the final image if return data is is True
        return img_copy
 

@app.route('/createCsv', methods=['GET', 'POST'])
def makeHeader():
    if request.method == 'POST':
        global filename
        filename = request.form['filename']

    with open(pathcsv+f'{filename}.csv', mode='a+', encoding='UTF-8') as csv_file:
        writer = csv.writer(csv_file)
        header = ['Ekspresi', 'Waktu']      
        writer.writerow(header)
        
    return redirect(url_for('recordData'))


def makeCsv():
    
    realTime = time.strftime('%H:%M:%S') 
    dataRow = [predicted_emotion, realTime]

    with open(pathcsv+f'{filename}.csv', mode='a+', encoding='UTF-8') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(dataRow)

    return redirect(url_for('recordData'))
    

def automation():
    namefile = filename

    time.sleep(1)
    visualizationData.visualization(namefile)

    return redirect(url_for('home'))

# @app.route('/genpercen/<filename>')
# def genPercen(filename):    
#     automations.makePieChartFromCsv(filename)            

#     return redirect(request.referrer)

# Routing Web

@app.route('/', methods = ['GET', 'POST'])
def home():        
    fNames = [x.name for x in os.scandir(pathcsv)]
    gNames = [x.name for x in os.scandir(visualPath)]

    return render_template('home.html', files = fNames, len = len(fNames), image = gNames, flname = gNames)

  
        
@app.route('/record', methods=['GET', 'POST'])
def recordData():
    fps=0      
    init_emotion() 
    cap = cv2.VideoCapture(0)

    while (True):
        start_time = time.time()
        success, frame= cap.read()   
        
        if not success:
            break                        

        image = cv2.flip(frame,1)        
        
        image = emotion(image, returndata=True, confidence = 0.8)
        makeCsv()
        

        cv2.putText(image, 'FPS: {:.2f}'.format(fps), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.8, (255, 20, 55), 1)
        cv2.imshow("Emotion Recognition",image)
        
        k = cv2.waitKey(1)
        fps= (1.0 / (time.time() - start_time))

        if k == ord('q'): # press 'q' to quit
            break                           

    cap.release()
    cv2.destroyAllWindows() 
    return automation()
           
    

@app.route('/grafik/<filename>')
def grafik(filename):   
    name = filename.upper().replace('.PNG', '')    
      
    return render_template('grafik.html', visualFile = filename, name = name)

@app.route('/about-us')
def aboutus():
    return render_template('aboutus.html')


@app.route('/del/<filename>')
def hapusfile(filename):
    filename = filename.replace('.png', '')
    flcsv = f'{pathcsv}{filename}.csv'
    flvisual = f'{visualPath}{filename}.png'
    
    if os.path.isfile(flcsv):
        os.remove(flcsv)
    else:
        pass
    
    if os.path.isfile(flvisual):
        os.remove(flvisual)
    else:
        pass       

    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
