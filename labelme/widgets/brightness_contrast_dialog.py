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

        self.slider_brightness = self._create_slider((-127, 127), 0)
        self.slider_contrast = self._create_slider((0, 200), 100)

        formLayout = QtWidgets.QFormLayout()
        formLayout.addRow(self.tr("Brightness"), self.slider_brightness)
        formLayout.addRow(self.tr("Contrast"), self.slider_contrast)
        self.setLayout(formLayout)

        assert isinstance(img, np.ndarray)
        self.img = img
        self.callback = callback

    def onNewValue(self):
        brightness = self.slider_brightness.value()
        contrast = self.slider_contrast.value() / 100.0

        img = self.img
        img = cv.convertScaleAbs(img, alpha=contrast, beta=brightness)

        img_data = utils.img_arr_to_data(img)
        qimage = QtGui.QImage.fromData(img_data)
        self.callback(qimage)

    def _create_slider(self, range, init_value):
        slider = QtWidgets.QSlider(Qt.Horizontal)
        slider.setRange(range[0], range[1])
        slider.setValue(init_value)
        slider.valueChanged.connect(self.onNewValue)
        return slider
