from matplotlib.pyplot import scatter
from paradox.NENV import *

widgets = import_widgets(__file__)

import numpy as np

class NodeBase(Node):
    color = '#6765eb'


class ImageShow(NodeBase):
    title = 'Imageshow'
    init_inputs = [
        NodeInputBP('x'),
    ]
    main_widget_class = widgets.ImageView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):

        img = self.input(0)

        if self.session.gui:
            self.main_widget().show_histogram(img)


class Histogram(NodeBase):
    title = 'histogram'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.HistogramView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)

# Bargraph block
class BarGraph(NodeBase):
    title = 'BarGraph'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.BarView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)


class Scatter(NodeBase):
    title = 'Scatter'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.ScatterView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)


class Stem(NodeBase):
    title = 'Stem'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.StemView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)

class PieChart(NodeBase):
    title = 'PieChart'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.PieView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x)


class Hist2D(NodeBase):
    title = 'Hist2D'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.Hist2dView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)


class Plot(NodeBase):
    title = 'Plot'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.PlotView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)


class CSVPlot(NodeBase):
    title = 'CSVPlot'
    init_inputs = [
        NodeInputBP('x'),
        NodeInputBP('y'),
    ]
    main_widget_class = widgets.PlotView
    main_widget_pos = 'below ports'

    def update_event(self, inp=-1):
        # hist = cv2.calcHist([self.input(0).img], [0], None, [256], [0,256])
        N = 50
        x = self.input(0)
        y = self.input(1)

        if self.session.gui:
            self.main_widget().show_histogram(x,y)


export_nodes(
    
  Histogram,
  BarGraph,
  PieChart,
  Scatter,
  Stem,
  Plot,
  Hist2D,
  ImageShow,

)