import os 
import pandas as pd
import librosa
import librosa.display
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers import Input, Flatten, Dropout, Activation
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models, Model, optimizers
import seaborn as sns
import keras
from keras.optimizers import Adam
import json
from keras.models import model_from_json
from tensorflow.keras.applications import vgg19
from keras.callbacks import EarlyStopping, ModelCheckpoint
import itertools
from sklearn.metrics import classification_report

PATH = 'C:/Users/e211/Desktop/Speech_Actor_01'

def main():
    train_data, test_data = produce_spec(PATH)
    data_tr, data_te, train_list, test_list, X_train, X_test, y_train, y_test = data_process(train_data, test_data)
    model, history, X_test, y_test = training(X_train, X_test, y_train, y_test)
    save_model(model)
    y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(y_pred,axis=1) 
    Y_true = np.argmax(y_test,axis=1)
    dict_characters = {0: 'angry', 1: 'calm', 2: 'disgust', 3: 'fearful', 4: 'happy', 5: 'neutral', 6: 'sad', 7: 'surprised'}
    confusion_mtx = confusion_matrix(Y_true, Y_pred_classes) 
    plot_confusion_matrix(confusion_mtx, classes = list(dict_characters.values())) 
    plt.show()
    print(classification_report(Y_true, Y_pred_classes, target_names = ['angry','calm','disgust','fearful','happy','neutral','sad','surprised']))

def produce_spec(path):
    #emotion = ['angry', 'neutral', 'sad','calm', 'disgust', 'happy', 'fearful', 'surprised']
    folder = os.listdir(path)
    save_dir = os.path.join(os.getcwd(), 'spec_pic')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    for index, y in enumerate(folder):
        X, sample_rate = librosa.load(os.path.join('/', path, y), sr = 22050 * 2, offset=0.5, duration = 2.5, res_type='kaiser_fast')
        sample_rate = np.array(sample_rate)
        D = np.abs(librosa.stft(X)) ** 2
        mel_spect = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=1024, hop_length=100)
        mel_spect = librosa.power_to_db(mel_spect, ref=np.max)
        librosa.display.specshow(mel_spect, y_axis='mel', fmax=20000, x_axis='time');
        plt.savefig(save_dir+ y +'.png')
    
    spec_folder = os.listdir(save_dir+'/'+'spec_pic')
    train_data, test_data = train_test_split(spec_folder, test_size=0.25, random_state=42)
    return train_data, test_data

def data_process(train_data, test_data):
    data_tr = []
    data_te = []
    train_list = [] #train的情緒類別
    test_list = []  #test的情緒類別
    
    for item in train_data:
        if item[6:-20]=='01':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('neutral')
        elif item[6:-20]=='02':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('calm')
        elif item[6:-20]=='03':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('happy')
        elif item[6:-20]=='04':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('sad')
        elif item[6:-20]=='05':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('angry')
        elif item[6:-20]=='06':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('fearful')
        elif item[6:-20]=='07':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('disgust')
        elif item[6:-20]=='08':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_tr.append(image)
            train_list.append('surprised')

    for item in test_data:
        if item[6:-20]=='01':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('neutral')
        elif item[6:-20]=='02':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('calm')
        elif item[6:-20]=='03':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('happy')
        elif item[6:-20]=='04':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('sad')
        elif item[6:-20]=='05':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('angry')
        elif item[6:-20]=='06':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('fearful')
        elif item[6:-20]=='07':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('disgust')
        elif item[6:-20]=='08':
            image=tf.keras.preprocessing.image.load_img(os.path.join('/', PATH, item), color_mode='rgb', target_size= (224,224))
            image=np.array(image)
            data_te.append(image)
            test_list.append('surprised')
            
    X_train = np.array(data_tr)
    y_train = np.array(train_list)

    X_test = np.array(data_te)
    y_test = np.array(test_list)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    lb = LabelEncoder()

    y_train = np_utils.to_categorical(lb.fit_transform(y_train))
    y_test = np_utils.to_categorical(lb.fit_transform(y_test))
    return data_tr, data_te, train_list, test_list, X_train, X_test, y_train, y_test

def training(X_train, X_test, y_train, y_test):
    img_height, img_width = 224,224
    conv_base = vgg19.VGG19(weights='imagenet', pooling='avg', include_top=False, input_shape = (img_width, img_height, 3))

    for layer in conv_base.layers[:12]:
        layer.trainable = False
    
    model=models.Sequential()
    model.add(conv_base)
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.01))
    model.add(Dense(8,activation='softmax'))
    model.summary()

    batch_size=16
    learning_rate = 5e-5
    epochs = 40
    checkpoint = ModelCheckpoint("vgg_19_classifier16.h5", monitor = 'val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(lr=learning_rate), metrics = ['acc'])

    history = model.fit(X_train, y_train,batch_size=batch_size,epochs=epochs,shuffle=True, validation_data=(X_test,y_test),callbacks=[checkpoint])
    
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('Augmented_Model_Accuracy.png')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    # plt.savefig('Augmented_Model_Loss.png')
    plt.show()

    print("Loss of the model is - " , model.evaluate(X_test,y_test)[0])
    print("Accuracy of the model is - " , model.evaluate(X_test,y_test)[1]*100 , "%")
    return model, history, X_test, y_test

def save_model(model):
    model_name = 'Emotion_Voice_Detection_Model.h5'
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    # Save model and weights
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('\nSaved trained model at %s ' % model_path)
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("saved_models/Emotion_Voice_Detection_Model.h5")
    print("Loaded model from disk")
    # evaluate loaded model on test data
    loaded_model.compile(loss='categorical_crossentropy', metrics=['acc'])

def plot_confusion_matrix(cm, classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
    plt.figure(figsize = (5,5))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.figure(figsize=(6,6))

main()