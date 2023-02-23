
# from cProfile import label
from pickle import HIGHEST_PROTOCOL, NONE
# from time import process_time_ns
from paradox.NENV import *
widgets = import_widgets(__file__)

# importing external python packages
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# importing logic file 
# from paradox.extension.Scikit_Learn import logic as ml
from paradox.extension.Scikit_Learn import predict as pd
from PySide2.QtWidgets import QVBoxLayout, QLabel, QPushButton, QWidget, QMainWindow, QApplication
from PySide2.QtCore import QTimer, QRunnable, Slot, Signal, QObject, QThreadPool
import sys
import time
import traceback
# import tensorflow_datasets as tfds


# from transformers import pipeline
# baseline cnn model for mnist

from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import threading
# import related to API

# import httpx

class WorkerSignals(QObject):
    '''
    Defines the signals available from a running worker thread.

    Supported signals are:

    finished
        No data

    error
        tuple (exctype, value, traceback.format_exc() )

    result
        object data returned from processing, anything

    progress
        int indicating % progress

    '''
    finished = Signal()
    error = Signal(tuple)
    result = Signal(object)
    progress = Signal(int)


class Worker(QRunnable):
    '''
    Worker thread

    Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

    :param callback: The function callback to run on this worker thread. Supplied args and
                     kwargs will be passed through to the runner.
    :type callback: function
    :param args: Arguments to pass to the callback function
    :param kwargs: Keywords to pass to the callback function

    '''

    def __init__(self, fn, *args, **kwargs):
        super(Worker, self).__init__()

        # Store constructor arguments (re-used for processing)
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.signals = WorkerSignals()

        # Add the callback to our kwargs
        self.kwargs['progress_callback'] = self.signals.progress

    @Slot()
    def run(self):
        '''
        Initialise the runner function with passed args, kwargs.
        '''

        # Retrieve args/kwargs here; and fire processing using them
        try:
            result = self.fn(*self.args, **self.kwargs)
        except:
            traceback.print_exc()
            exctype, value = sys.exc_info()[:2]
            self.signals.error.emit((exctype, value, traceback.format_exc()))
        else:
            self.signals.result.emit(result)  # Return the result of the processing
        finally:
            self.signals.finished.emit()  # Done


# BaseNode class 

class NodeBase(Node):
    color = '#00a6ff'


# load csv file
class ReadCSV(NodeBase):
    """Reads an image from a file"""

    title = 'Read CSV'
    input_widget_classes = {
        'choose file IW': widgets.ChooseFileInputWidget
    }
    init_inputs = [
        NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
    ]
    init_outputs = [
        NodeOutputBP('data')
    ]

    def __init__(self, params):
        super().__init__(params)

        self.image_filepath = ''

    def view_place_event(self):
        self.input_widget(0).path_chosen.connect(self.path_chosen)
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):
        if self.image_filepath == '':
            return

        try:
            self.set_output_val(0, (pd.read_csv(self.image_filepath)))
        except Exception as e:
            print(e)

    def get_state(self):
        data = {'image file path': self.image_filepath}
        return data

    def set_state(self, data, version):
        self.path_chosen(data['csv file path'])
        # self.image_filepath = data['image file path']

    def path_chosen(self, file_path):
        self.image_filepath = file_path
        self.update()


class ReadImage(NodeBase):
    """Reads an image from a file"""

    title = 'Load Model'
    input_widget_classes = {
        'choose file IW': widgets.ChooseFileInputWidget
    }
    init_inputs = [
        NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
    ]
    init_outputs = [
        NodeOutputBP('data')
    ]

    def __init__(self, params):
        super().__init__(params)

        self.image_filepath = ''

    def view_place_event(self):
        self.input_widget(0).path_chosen.connect(self.path_chosen)
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):
        if self.image_filepath == '':
            return

        try:
            self.set_output_val(0, (self.image_filepath))
        except Exception as e:
            print(e)

    def get_state(self):
        data = {'image file path': self.image_filepath}
        return data

    def set_state(self, data, version):
        self.path_chosen(data['csv file path'])
        # self.image_filepath = data['image file path']

    def path_chosen(self, file_path):
        self.image_filepath = file_path
        self.update()
# Read folder
class ReadFolder(NodeBase):
    """Reads an image from a file"""

    title = 'Read Folder'
    input_widget_classes = {
        'choose file IW': widgets.ChooseFolderInputWidget
    }
    init_inputs = [
        NodeInputBP('f_path', add_data={'widget name': 'choose file IW', 'widget pos': 'besides'})
    ]
    init_outputs = [
        NodeOutputBP('data')
    ]

    def __init__(self, params):
        super().__init__(params)

        self.image_filepath = ''

    def view_place_event(self):
        self.input_widget(0).path_chosen.connect(self.path_chosen)
        # self.main_widget_message.connect(self.main_widget().show_path)

    def update_event(self, inp=-1):
        if self.image_filepath == '':
            return

        try:
            self.set_output_val(0, self.image_filepath)
        except Exception as e:
            print(e)

    def get_state(self):
        data = {'image file path': self.image_filepath}
        return data

    def set_state(self, data, version):
        self.path_chosen(data['csv file path'])
        # self.image_filepath = data['image file path']

    def path_chosen(self, file_path):
        self.image_filepath = file_path
        self.update()

class ChooseData(NodeBase):
    title = 'Choose Data'
    version = 'v0.1'

    data_codes = {
        'load_diabetes' : datasets.load_diabetes(return_X_y=True),
        'load_mnist' : datasets.load_digits(return_X_y=True),
        'load_breast_cancer' : datasets.load_breast_cancer(return_X_y=True),
        'load_iris' : datasets.load_iris(return_X_y=True),
        'load_wine' : datasets.load_wine(return_X_y=True),
      
    } 



    init_inputs = [
        NodeInputBP(type_='exec'),
        NodeInputBP('data', dtype=dtypes.Choice(list(data_codes.keys())[0], list(data_codes.keys()))),
    ]

    init_outputs = [
        NodeOutputBP('x_train'),
        NodeOutputBP('y_train'),
        NodeOutputBP('x_test'),
        NodeOutputBP('y_test'),
    ]


    def update_event(self, inp=-1):
        # diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)
        x_train , y_train = self.data_codes[self.input(1)]
      
        # print(x_train)
        # print(y_train)

        x_train = x_train[:, np.newaxis, 2]

        X_train = x_train[:-20]
        X_test = x_train[-20:]

        Y_train = y_train[:-20]
        Y_test = y_train[-20:]

        self.set_output_val(0,X_train)
        self.set_output_val(2,X_test)
        self.set_output_val(1,Y_train)
        self.set_output_val(3,Y_test)
# import numba as nb
# Predication block
class Predict(NodeBase):
    title = 'Predict'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP('Image'),
        NodeInputBP('Model'),
        NodeInputBP('label'),
    ]


    init_outputs = [
        NodeOutputBP('Prediction'),

    ]

    def update_event(self, inp=-1):
        image_path = self.input(0)
        model_path = self.input(1)
        dataset_path = self.input(2)
        
        predicted = pd.predict(image_path, model_path,dataset_path)

        self.set_output_val(0,predicted)

# Predication block
# class Tensorflowmodel(NodeBase):
#     title = 'Model Traning'
#     version = 'v0.1'
#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('data'),
#     ]

#     init_outputs = [
#         NodeOutputBP('predication'),  
#     ]

#     def progress_fn(self, n):  
    
#         # trainX, trainY, testX, testY = ml.load_dataset()
#         # trainX, testX = ml.prep_pixels(trainX, testX)
#         # model = ml.define_model()
#         # model.fit(trainX, trainY, epochs=3, batch_size=10, validation_data=(testX, testY), verbose=2)
        
#         # acc,history = ml.train_model()

#         print("%d%% done" % n)
#         # print(trainX)
#         # print(trainY)
 
#     # @nb.njit           
#     def execute_this_fn(self, progress_callback):
#         print("Starting your ml training...")
#         ml.train_model()
#         print("Done")

#     def print_output(self, s):
#         print(s)

#     def thread_complete(self):

#         print("THREAD COMPLETE!")
    
#     # @nb.njit
#     def myfunc(self):
#         # Implementing the threading 
#         worker = Worker(self.execute_this_fn) # Any other args, kwargs are passed to the run function
#         worker.signals.result.connect(self.print_output)
#         worker.signals.finished.connect(self.thread_complete)
#         worker.signals.progress.connect(self.progress_fn)
        
#         # Execute
#         self.threadpool.start(worker)
        
#     # @nb.njit    
#     def update_event(self, inp=-1):
#         # printing the number of threads
#         self.threadpool = QThreadPool()
#         print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
#         self.myfunc()
        
        
import pickle

# Build the model 
class Regression_Model(NodeBase):
    title = 'Regression Model'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(type_='exec'),
        NodeInputBP('x_train'),
        NodeInputBP('y_train'),
    ]
    init_outputs = [
        NodeOutputBP('model'),  
    ]


    def update_event(self, inp=-1): 

        print(" traning is started.. ")
        
        x_train = self.input(1)
        y_train = self.input(2)

        model = linear_model.LinearRegression()
        model = model.fit(x_train,y_train)

        # save
        joblib.dump(model, "model.pkl") 

        # load
        clf2 = joblib.load("model.pkl")

        self.set_output_val(0,clf2)

# Logistic Regression
class Logistic_Model(NodeBase):
    title = 'Logistic Model'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(type_='exec'),
        NodeInputBP('x_train'),
        NodeInputBP('y_train'),
    ]
    init_outputs = [
        NodeOutputBP('model'),  
    ]


    def update_event(self, inp=-1): 

        print(" traning is started.. ")
        
        x_train = self.input(1)
        y_train = self.input(2)


        model = linear_model.LogisticRegression()
        model = model.fit(x_train,y_train)

        # save
        joblib.dump(model, "model.pkl") 

        # load
        clf2 = joblib.load("model.pkl")

        self.set_output_val(0,clf2)


# class APIGet(NodeBase):
#     title = 'API Get'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('URL'),
#     ]

#     init_outputs = [
#         NodeOutputBP('response'),
#     ]

#     def update_event(self, inp=-1):
#         url = self.input(1)
#         response = httpx.get(url)
#         self.set_output_val(0,response)

# class APIPost(NodeBase):
#     title = 'API Post'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('URL'),
#     ]

#     init_outputs = [
#         NodeOutputBP('response'),
#     ]

#     def update_event(self, inp=-1):
#         url = self.input(1)
#         response = httpx.post(url)
#         self.set_output_val(0,response)



# class APIPut(NodeBase):
#     title = 'API Put'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('URL'),
#     ]

#     init_outputs = [
#         NodeOutputBP('response'),
#     ]

#     def update_event(self, inp=-1):
#         url = self.input(1)
#         response = httpx.put(url)
#         self.set_output_val(0,response)


# class APIOption(NodeBase):
#     title = 'API Option'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('URL'),
#     ]

#     init_outputs = [
#         NodeOutputBP('response'),
#     ]

#     def update_event(self, inp=-1):
#         url = self.input(1)
#         response = httpx.options(url)
#         self.set_output_val(0,response)


# class APIDelete(NodeBase):
#     title = 'API Delete'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('URL'),
#     ]

#     init_outputs = [
#         NodeOutputBP('response'),
#     ]

#     def update_event(self, inp=-1):
#         url = self.input(1)
#         response = httpx.delete(url)
#         self.set_output_val(0,response)

# class APIHead(NodeBase):
#     title = 'API Head'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('URL'),
#     ]

#     init_outputs = [
#         NodeOutputBP('response'),
#     ]

#     def update_event(self, inp=-1):
#         url = self.input(1)
#         response = httpx.head(url)
#         self.set_output_val(0,response)


# class ToText(NodeBase):
#     title = 'convert to text'
#     version = 'v0.1'


#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('response'),
#     ]

#     init_outputs = [
#         NodeOutputBP('result'),
#     ]

#     def update_event(self, inp=-1):
#         response = self.input(1)
#         result = response.text()
#         self.set_output_val(0,result)


# importing flet package 
# import flet
# from flet import Page, Text
# import threading
# import ray
# class Alien(NodeBase):
#     title = 'alien'
#     version = 'v0.1'
#     init_inputs = [
#         NodeInputBP(type_='exec'),
#         NodeInputBP('name'),
#     ]

#     init_outputs = [
#         NodeOutputBP('predication'),  
#     ]


 
#     def myapp(self,page: Page):
#         name = self.input(1)
#         t = Text(value=name, color="green")
#         page.controls.append(t)
#         page.update()
        
#     def progress_fn(self, n):
#         # flet.app(target=self.myapp)
                
#         print("%d%% done" % n)
         
                     
#     def execute_this_fn(self, progress_callback):
#         print("Starting your ml training...")
#         # ml.train_model()   
#         print("Done")

#     def print_output(self, s):
#         print(s)
        
#     @ray.remote
#     def thread_complete(self):
        
#         t1 = threading.Thread(target=flet.app(target=self.myapp,view=flet.WEB_BROWSER))
#         t1.start()

#         print("THREAD COMPLETE!")
    
#     @ray.remote
#     def myfunc(self):
#         # Implementing the threading 
#         worker = Worker(self.execute_this_fn) # Any other args, kwargs are passed to the run function
#         worker.signals.result.connect(self.print_output)
#         worker.signals.finished.connect(self.thread_complete.remote())
#         worker.signals.progress.connect(self.progress_fn)
        
#         # Execute
#         self.threadpool.start(worker)
        
#     def update_event(self, inp=-1):
#         # printing the number of threads
#         self.threadpool = QThreadPool()
#         print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
#         self.myfunc.remote()

export_nodes(
    # register the nodes in the extension
    ReadCSV,
    ReadFolder,
    Predict,
    ChooseData,
    ReadImage,

    # Models
    Regression_Model,
    Logistic_Model,

    #  API blocks
    # APIGet,
    # APIDelete,
    # APIHead,
    # APIPut,
    # APIPost,
    # APIOption,

    # ToText,
    # Alien,
    # Tensorflowmodel,


)