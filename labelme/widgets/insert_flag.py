from qtpy import QT_VERSION
from qtpy import QtWidgets


QT5 = QT_VERSION[0] == "5"


class InsertFlagWidget(QtWidgets.QWidget):
    def __init__(self, callback, parent=None):
        super(InsertFlagWidget, self).__init__(parent)

        layout = QtWidgets.QFormLayout()
        self.button = QtWidgets.QPushButton("Insert Flag", self)
        self.button.clicked.connect(self.onPushButtonClicked)

        self.lineedit = QtWidgets.QLineEdit()
        layout.addRow(self.button, self.lineedit)

        self.setLayout(layout)
        self.callback = callback

    def onPushButtonClicked(self):
        self.callback(self.lineedit.text())
        self.lineedit.setText("")
