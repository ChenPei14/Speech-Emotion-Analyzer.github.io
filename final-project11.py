import tkinter as tk
from tkinter import *
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import os 
import librosa
import librosa.display
from keras.models import model_from_json
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import tensorflow as tf
import sys

class Application(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.pack()
        self.create_widgets()

    def create_widgets(self):
        self.my_label1 = tk.Label(self)
        self.my_label1["text"] = 'Load wav file'
        self.my_label1['font'] = ("Times New Roman", 18, 'bold')
        self.my_label1.pack(anchor=tk.CENTER)
        
        self.my_button1 = tk.Button(self)
        self.my_button1["text"] = "Browse"
        self.my_button1["command"] = self.open_file
        self.my_button1['font'] = ("Times New Roman", 16, 'bold')
        self.my_button1.pack()
        
        self.entry_filename = tk.Entry(self)
        self.entry_filename['width'] = 30
        self.entry_filename['font'] = ("標楷體", 10, 'bold')
        self.entry_filename.pack()
        
        self.my_button2 = tk.Button(self)
        self.my_button2["text"] = "輸出"
        self.my_button2["command"] = self.print_file
        self.my_button2['font'] = ("標楷體", 16, 'bold')
        self.my_button2.pack()
        
        self.my_button3 = tk.Button(self)
        self.my_button3["text"] = "Begin"
        self.my_button3["command"] = self.prediction
        self.my_button3['font'] = ("Times New Roman", 16, 'bold')
        self.my_button3.pack()
        
        self.cv1 = tk.Canvas(self)
        self.cv1['width'] = 480
        self.cv1['height'] = 200
        self.cv1['bg'] = 'white'
        self.cv1.create_rectangle(20,15,470,180)
        self.cv1.create_text(250, 80, font=("標楷體", 16, 'bold'), text='顯示載入音檔\n 的wavform')
        self.cv1.pack()
        
        self.cv2 = tk.Canvas(self)
        self.cv2['width'] = 480
        self.cv2['height'] = 200
        self.cv2['bg'] = 'white'
        self.cv2.create_rectangle(20,15,470,180)
        self.text1=self.cv2.create_text(250, 80, font=("標楷體", 16, 'bold'), text='顯示辨識後\n  的結果')
        self.cv2.pack()
        
        self.quit = tk.Button(self, text="QUIT", command=self.master.destroy)
        self.quit.pack(side="bottom")
        
    def open_file(self):
        filename = filedialog.askopenfilename(title='Open Wav File', filetypes=[('wav', '*.wav')])
        self.entry_filename.insert('insert', filename)

    def print_file(self):
        save_dir = os.path.join(os.getcwd(), 'out_pic')
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        
        real_filepath = self.entry_filename.get()  #用get提取entry中的內容
        #filename = real_filepath[22:]
        data, sampling_rate = librosa.load(real_filepath)
        fig = plt.figure(figsize=(15, 5))
        librosa.display.waveplot(data, sr=sampling_rate) #關閉才會顯示(要讓它顯示再tkinterk)
        fig.savefig(os.path.join(save_dir, 'outwav.png'))
        global image
        global img
        image = Image.open(os.path.join(save_dir, 'outwav.png')) 
        (x, y)=image.size
        re_x, re_y = 480, 200
        out = image.resize((re_x,re_y),Image.ANTIALIAS)
        img = ImageTk.PhotoImage(out)
        self.cv1.create_image(0,0,anchor=NW,  image=img)
        return real_filepath, save_dir

    def prediction(self): 
        real_filepath, save_dir = self.print_file()
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(os.path.join(os.getcwd(), "saved_models/Emotion_Voice_Detection_Model.h5"))
        print("Loaded model from disk")
        loaded_model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
        
        X, sample_rate = librosa.load(real_filepath, sr = 22050 * 2, offset=0.5, duration = 2.5, res_type='kaiser_fast')
        sample_rate = np.array(sample_rate)
        mel_spect = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024, hop_length=100)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=20000, x_axis='time');
        plt.savefig(os.path.join(save_dir, 'outspec.png'))
    
        image=tf.keras.preprocessing.image.load_img(os.path.join(save_dir, 'outspec.png'), color_mode='rgb', target_size= (224,224))
        image=np.array(image)
        
        livedf2 = image
        twodim= np.expand_dims(livedf2, axis=0)
        livepreds = loaded_model.predict(twodim, batch_size=16, verbose=1)
        
        livepreds1=livepreds.argmax(axis=1) #返回最大值，橫著比較
        liveabc = livepreds1.astype(int).flatten()
        test_labels = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
        lb = LabelEncoder()
        result = lb.fit_transform(test_labels)
        livepredictions = (lb.inverse_transform((liveabc)))
        print(livepredictions)
        self.cv2.itemconfig(self.text1, font=("Times New Roman", 25, 'bold'), text= livepredictions)
        return livepredictions

root = tk.Tk()
root.title('Load wav file')
root.geometry('600x700')
app = Application(master=root)
app.mainloop()        