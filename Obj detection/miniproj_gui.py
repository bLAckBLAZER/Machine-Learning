from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import cv2

root=Tk()
root.title("Object Detection")
root.geometry("600x600+400+200")

path = Entry(root,width=50,bd=5)
path.pack()

#To select the input image
def browsefunc():
    path.delete(0,END)
    filename = filedialog.askopenfilename()
    path.insert(END,filename)

browsebutton = Button(root, text="Browse", command=browsefunc)
browsebutton.pack()

def resizeimg(imgpath):

    #Use cv2 to read and resize the image
    pic = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED)
    #scale_percent = 20 # percent of original size
    #width = int(pic.shape[1] * scale_percent / 100)
    #height = int(pic.shape[0] * scale_percent / 100)
    width = 700
    height = 500

    dim = (width, height)
    # resize image
    resized = cv2.resize(pic, dim, interpolation = cv2.INTER_AREA)
    resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) #because when we do cv2.read(), it converts the color of image and thus we need to change it back
    resized_pil = Image.fromarray(resized) #convert cv2 image obj to a PIL image obj
    return resized_pil

#Displaying image
def openimg():
    #resize Image
    imgpath= path.get()
    inputimg = resizeimg(imgpath) #we get resized image in PIL format

    #inputimg = Image.open(inputimg) #takes input image in any format
    inputimg = ImageTk.PhotoImage(inputimg) #converts the image into tkinter supported format

    panel.configure(image=inputimg)
    panel.image = inputimg

    #panel.pack(side = "bottom", fill = "both", expand = "yes")
openbutton = Button(root,text='Open', command=openimg)
openbutton.pack()

def detectobj():
    
   
    import time
    import numpy as np
    from model.yolo_model import YOLO

    def process_image(img):

        image = cv2.resize(img, (416, 416),interpolation=cv2.INTER_CUBIC)
        image = np.array(image, dtype='float32')
        image /= 255.
        image = np.expand_dims(image, axis=0)

        return image


    def get_classes(file):
   
        with open(file) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]

        return class_names



    def draw(image, boxes, scores, classes, all_classes):
  
        for box, score, cl in zip(boxes, scores, classes):
            x, y, w, h = box

            top = max(0, np.floor(x + 0.5).astype(int))
            left = max(0, np.floor(y + 0.5).astype(int))
            right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
            bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

            cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
            cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),(top, left - 6),cv2.FONT_HERSHEY_SIMPLEX,0.6, (0, 0, 255), 1,cv2.LINE_AA)

            print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
            print('box coordinate x,y,w,h: {0}'.format(box))

        print()
  


    def detect_image(image, yolo, all_classes):
   
        pimage = process_image(image)

        start = time.time()
        boxes, classes, scores = yolo.predict(pimage, image.shape)
        end = time.time()

        print('time: {0:.2f}s'.format(end - start))

        if boxes is not None:
            draw(image, boxes, scores, classes, all_classes)

        return image


    yolo = YOLO(0.6, 0.5)
    file = 'data/coco_classes.txt'
    all_classes = get_classes(file)




    f = path.get()
    f = f.split('.')
    f=f[0]+'_edited.jpg'
    #path = 'C:/Users/OMKAR/YOLOv3/images/'+f
    image = cv2.imread(path.get())
    image = detect_image(image, yolo, all_classes)
    cv2.imwrite(f, image)

    imgpath=  f
    inputimg = resizeimg(imgpath) #we get resized image in PIL format

    #inputimg = Image.open(inputimg) #takes input image in any format
    inputimg = ImageTk.PhotoImage(inputimg) #converts the image into tkinter supported format

    panel.configure(image=inputimg)
    panel.image = inputimg



    
    
    
    

detectbutton = Button(root, text= 'Detect', command= detectobj)
detectbutton.pack()
panel = Label(root)
panel.pack()







root.mainloop()

