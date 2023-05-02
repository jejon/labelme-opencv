import cv2 as cv
import numpy as np
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy import QtWidgets

from .. import utils


class BrightnessContrastDialog(QtWidgets.QDialog):
    def __init__(self, img, callback, parent=None):
        super(BrightnessContrastDialog, self).__init__(parent)
        self.setModal(True)
        self.setWindowTitle("Brightness/Contrast")

        self.init_value_brightness = 0
        self.slider_brightness = \
            self._create_slider((-127, 127), self.init_value_brightness)
        self.init_value_contrast = 100
        self.slider_contrast = \
            self._create_slider((0, 200), self.init_value_contrast)
        self.buttonHist = QtWidgets.QPushButton("Histogram equalization", self)
        self.buttonHist.setCheckable(True)
        self.buttonHist.setChecked(False)
        self.buttonHist.setAutoDefault(False)
        self.buttonHist.clicked.connect(self.onHistogramEqualization)
        self.buttonNorm = QtWidgets.QPushButton("Normalization", self)
        self.buttonNorm.setCheckable(True)
        self.buttonNorm.setChecked(False)
        self.buttonNorm.setAutoDefault(False)
        self.buttonNorm.clicked.connect(self.onNormalization)
        self.buttonReset = QtWidgets.QPushButton("Reset", self)
        self.buttonReset.setAutoDefault(False)
        self.buttonReset.clicked.connect(self.onReset)

        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.tr("Brightness"), self.slider_brightness)
        formLayout.addRow(self.tr("Contrast"), self.slider_contrast)
        formLayout.addRow(self.tr(""), self.buttonHist)
        formLayout.addRow(self.tr(""), self.buttonNorm)
        formLayout.addRow(self.tr(""), self.buttonReset)
        self.setLayout(formLayout)

        assert isinstance(img, np.ndarray)
        self.img = img
        self.img_working = img
        self.callback = callback

    def onNewValue(self):
        brightness = self.slider_brightness.value()
        contrast = self.slider_contrast.value() / 100.0

        img = self.img_working
        img = cv.convertScaleAbs(img, alpha=contrast, beta=brightness)
        img_data = utils.img_arr_to_data(img)
        qimage = QtGui.QImage.fromData(img_data)
        self.callback(qimage)

    def onNormalization(self):
        self.slider_brightness.setValue(0)
        if self.buttonNorm.isChecked():
            img = self.img_working
            if img.dtype == np.uint16:
                img = cv.normalize(img, None, 0, 65535, cv.NORM_MINMAX)
            else:
                img = cv.normalize(img, None, 0, 255, cv.NORM_MINMAX)
            self.img_working = img
            img_data = utils.img_arr_to_data(img)
            qimage = QtGui.QImage.fromData(img_data)
            self.callback(qimage)
        else:
            self.onReset()

    def onHistogramEqualization(self):
        if self.buttonHist.isChecked():
            img = self.img_working
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            if img.dtype == np.uint16:
                img = (img // 256).astype(np.uint8)
            clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
            self.img_working = img
            img_data = utils.img_arr_to_data(img)
            qimage = QtGui.QImage.fromData(img_data)
            self.callback(qimage)
        else:
            self.onReset()

    def onReset(self):
        self.img_working = self.img
        self.slider_brightness.valueChanged.disconnect()
        self.slider_brightness.setValue(self.init_value_brightness)
        self.slider_brightness.valueChanged.connect(self.onNewValue)
        self.slider_contrast.valueChanged.disconnect()
        self.slider_contrast.setValue(self.init_value_contrast)
        self.slider_contrast.valueChanged.connect(self.onNewValue)

        self.buttonHist.setChecked(False)
        self.buttonNorm.setChecked(False)
        self.buttonReset.setDefault(False)

        img_data = utils.img_arr_to_data(self.img)
        qimage = QtGui.QImage.fromData(img_data)
        self.callback(qimage)

    def _create_slider(self, range, init_value):
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(range[0], range[1])
        slider.setValue(init_value)
        slider.valueChanged.connect(self.onNewValue)
        return slider
