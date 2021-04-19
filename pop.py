# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'pop.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore
from pyqtgraph import MultiPlotWidget


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1164, 638)
        self.pop_graph = MultiPlotWidget(Form)
        self.pop_graph.setGeometry(QtCore.QRect(40, 40, 1101, 561))
        self.pop_graph.setObjectName("pop_graph")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
