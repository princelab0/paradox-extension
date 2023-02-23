from paradox.NWENV import *
from qtpy.QtWidgets import QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTextEdit
from qtpy.QtGui import QImage, QPixmap, QFont
from qtpy.QtCore import Signal, QSize, QTimer

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
matplotlib.use('Qt5Agg')
import os 
import numpy as np
from mpl_toolkits import mplot3d




class ImageView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        # self.figure = plt.Figure(figsize=(255, 10))
        self.figure = plt.imshow(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, img):

        self.figure.clear()
        self.figure(img)
        self.canvas.draw()

# widget for the hitogram
class HistogramView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist(x, bins=8, linewidth=0.5, edgecolor="white")
        self.canvas.draw()

class PlotView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x,y)
        self.canvas.draw()


class StemView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.stem(x,y)
        self.canvas.draw()


class PlotView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.plot(x,y)
        self.canvas.draw()


class BarView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(x,y, width=1, edgecolor="white", linewidth=0.7)
        self.canvas.draw()

class StepView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.step(x,y)
        self.canvas.draw()  


class ScatterView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(x,y, width=1,vmin=0, vmax=100)
        self.canvas.draw()


class PieView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.pie(x, colors="blue", radius=3, center=(4, 4),
         wedgeprops={"linewidth": 1, "edgecolor": "white"}, frame=True)
        self.canvas.draw()


class Hist2dView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist2d(x,y)
        self.canvas.draw()



class CSVGraphView(MWB, QWidget):
    def __init__(self, params):
        MWB.__init__(self, params)
        QWidget.__init__(self)

        self.figure = plt.Figure(figsize=(255, 10))
        self.canvas = FigureCanvasQTAgg(self.figure)
        # self.ax = self.canvas.figure.subplots()

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.canvas)

        self.setFixedWidth(400)
        self.setFixedHeight(300)

    def show_histogram(self, x,y):

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.hist2d(x,y)
        self.canvas.draw()


# exporting the widgets
export_widgets(
    HistogramView,
    Hist2dView,
    PlotView,
    BarView,
    PieView,
    StemView,
    ScatterView,
    CSVGraphView,
    ImageView,
  
)