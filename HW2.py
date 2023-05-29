import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsScene
from PyQt5.QtCore import Qt, QThread, QTimer

import matplotlib

matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import pyplot as plt

import numpy as np

import os

import random

import Ui_RBFN
import simple_playground as sp



def drawInit():
    
    for line in Line:
        print('Line:', line.p1.x, line.p1.y, line.p2.x, line.p2.y)
    #----------------------------------------------------------------
    F1 = PlotCanvas(width=1, height=4, dpi=100)

    width, height = ui.graphicsView.width(), ui.graphicsView.height()
    F1.resize(width, height)

    scene =  QGraphicsScene()
    #----------------------------------------------------------------

    pos1 = [0, 0]
    pos2 = [0, 1]
    # draw
    F1.drawPlayground()
    F1.drawCar(pos1, pos2)
    
    
    scene.addWidget(F1)
    ui.graphicsView.setScene(scene)
    

def RBFNLearn():
    global carInfo
    global count

    count = 0
    k = ui.knum.value()
    k=10
    if ui.radio4d.isChecked():
        carInfo = sp.run_example(k)
    else:
        carInfo = sp.run_example6d(k)
    
    ui.Walk.setEnabled(True)
    print('carInfo', carInfo)

timer = QTimer()   
def nextStep():
    global count
    global carInfo
    count = 0
    if count < len(carInfo):
        #print('$$')
        timer.timeout.connect(run)
        timer.start(250)
    
 


    # nowpos = carInfo[count]
    # if count == len(carInfo) - 1:
    #     nextpos = carInfo[count]
    # else:
    #     nextpos = carInfo[count + 1]
    # F1.drawPlayground()
    # F1.drawCar(nowpos[:2], nextpos[:2])
    
    
    # scene.addWidget(F1)
    # ui.graphicsView.setScene(scene)
    # ui.label.setText(str(nowpos[2:]))
    # if count < len(carInfo) - 1:
    #     count += 1

def run():
    global count
    global carInfo
    #print('##')
    F1 = PlotCanvas(width=1, height=4, dpi=100)

    width, height = ui.graphicsView.width(), ui.graphicsView.height()
    F1.resize(width, height)
    scene =  QGraphicsScene()

    nowpos = carInfo[count]
    if count == len(carInfo) - 1:
        nextpos = carInfo[count]
    else:
        nextpos = carInfo[count + 1]
    F1.drawPlayground()
    F1.drawCar(nowpos[:2], nextpos[:2])
    
    
    scene.addWidget(F1)
    ui.graphicsView.setScene(scene)
    ui.label.setText(str(nowpos[2:]))
    if count < len(carInfo) - 1:
        count += 1
    else:
        timer.stop()

def selectTrack():
    global carInfo
    count = 0
    fileName = ui.comboBox.currentText()
    Info = []
    with open(fileName, 'r') as file:
        for f in file.readlines():
            temp = list(map(float, f.split()))
            Info.append(temp)
    p = sp.Playground()
    carInfo = []
    for i in Info:
        state = p.step(i[-1])
        carInfo.append([p.car.getPosition('center').x, p.car.getPosition('center').y] + state)
    print(carInfo)
    


class PlotCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        
        super(PlotCanvas, self).__init__(self.fig)
        self.ax = self.figure.add_subplot(111) # 2d
    def drawPlayground(self):
        global carInfo
        global count

        
        for line in Line:
            self.ax.plot((line.p1.x, line.p2.x), (line.p1.y, line.p2.y), color = 'black')


        draw_rectangle = plt.Rectangle((18, 37), 12, 3, facecolor='orange', fill=True)

        self.ax.set_aspect(1)
        self.ax.add_artist(draw_rectangle)

        self.ax.plot((-6, 6), (0, 0), color = 'orange')

        prevtrack = [0, 0]
        trackcount = 0
        for track in carInfo:
            if trackcount == count+1:
                break
            self.ax.plot((prevtrack[0], track[0]), (prevtrack[1], track[1]), color = 'blue')
            prevtrack = track[:2]

            trackcount += 1

        self.draw()

    def drawCar(self, pos1, pos2):
        
        draw_circle = plt.Circle((pos1[0], pos1[1]), 3, fill=False)

        self.ax.set_aspect(1)
        self.ax.add_artist(draw_circle)

        self.ax.quiver(pos1[0], pos1[1], pos2[0] - pos1[0], pos2[1] - pos1[1], 20)

        self.draw()

playground = sp.Playground()
Line = playground.lines
carInfo = []
count = 0
if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_RBFN.Ui_MainWindow()
    ui.setupUi(MainWindow)

    ui.graphicsView.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    ui.graphicsView.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    
    ui.Train.clicked.connect(RBFNLearn)
    ui.Walk.clicked.connect(nextStep)
    ui.Track.clicked.connect(selectTrack)
    
    #ui.Walk.setEnabled(False)
    drawInit()
    MainWindow.show()
    sys.exit(app.exec_())
