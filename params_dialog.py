# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\params_dialog.ui'
#
# Created by: PyQt5 UI code generator 5.9
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(449, 227)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(100, 190, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 431, 181))
        self.groupBox.setObjectName("groupBox")
        self.layoutWidget = QtWidgets.QWidget(self.groupBox)
        self.layoutWidget.setGeometry(QtCore.QRect(10, 20, 411, 131))
        self.layoutWidget.setObjectName("layoutWidget")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.layoutWidget)
        self.gridLayout_2.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.vpp_bound_upper = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.vpp_bound_upper.setMaximum(200.0)
        self.vpp_bound_upper.setSingleStep(0.01)
        self.vpp_bound_upper.setProperty("value", 200.0)
        self.vpp_bound_upper.setObjectName("vpp_bound_upper")
        self.gridLayout_2.addWidget(self.vpp_bound_upper, 2, 1, 1, 1)
        self.upper_bounds_label = QtWidgets.QLabel(self.layoutWidget)
        self.upper_bounds_label.setObjectName("upper_bounds_label")
        self.gridLayout_2.addWidget(self.upper_bounds_label, 2, 0, 1, 1)
        self.lower_bounds_label = QtWidgets.QLabel(self.layoutWidget)
        self.lower_bounds_label.setObjectName("lower_bounds_label")
        self.gridLayout_2.addWidget(self.lower_bounds_label, 1, 0, 1, 1)
        self.vpp_bound_lower = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.vpp_bound_lower.setPrefix("")
        self.vpp_bound_lower.setMaximum(200.0)
        self.vpp_bound_lower.setSingleStep(0.01)
        self.vpp_bound_lower.setObjectName("vpp_bound_lower")
        self.gridLayout_2.addWidget(self.vpp_bound_lower, 1, 1, 1, 1)
        self.constant_bound_lower = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.constant_bound_lower.setMaximum(0.2)
        self.constant_bound_lower.setSingleStep(0.01)
        self.constant_bound_lower.setObjectName("constant_bound_lower")
        self.gridLayout_2.addWidget(self.constant_bound_lower, 1, 4, 1, 1)
        self.frequency_bound_lower = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.frequency_bound_lower.setMaximum(1000.0)
        self.frequency_bound_lower.setObjectName("frequency_bound_lower")
        self.gridLayout_2.addWidget(self.frequency_bound_lower, 1, 3, 1, 1)
        self.constant_bound_initial = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.constant_bound_initial.setMaximum(0.2)
        self.constant_bound_initial.setSingleStep(0.01)
        self.constant_bound_initial.setObjectName("constant_bound_initial")
        self.gridLayout_2.addWidget(self.constant_bound_initial, 3, 4, 1, 1)
        self.decay_time_bound_upper = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.decay_time_bound_upper.setMaximum(100.0)
        self.decay_time_bound_upper.setSingleStep(0.01)
        self.decay_time_bound_upper.setProperty("value", 100.0)
        self.decay_time_bound_upper.setObjectName("decay_time_bound_upper")
        self.gridLayout_2.addWidget(self.decay_time_bound_upper, 2, 2, 1, 1)
        self.decay_time_bound_lower = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.decay_time_bound_lower.setMaximum(100.0)
        self.decay_time_bound_lower.setSingleStep(0.01)
        self.decay_time_bound_lower.setObjectName("decay_time_bound_lower")
        self.gridLayout_2.addWidget(self.decay_time_bound_lower, 1, 2, 1, 1)
        self.frequency_bound_initial = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.frequency_bound_initial.setMaximum(1000.0)
        self.frequency_bound_initial.setSingleStep(1.0)
        self.frequency_bound_initial.setProperty("value", 300.0)
        self.frequency_bound_initial.setObjectName("frequency_bound_initial")
        self.gridLayout_2.addWidget(self.frequency_bound_initial, 3, 3, 1, 1)
        self.frequency_bound_upper = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.frequency_bound_upper.setMaximum(1000.0)
        self.frequency_bound_upper.setProperty("value", 1000.0)
        self.frequency_bound_upper.setObjectName("frequency_bound_upper")
        self.gridLayout_2.addWidget(self.frequency_bound_upper, 2, 3, 1, 1)
        self.constant_bound_upper = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.constant_bound_upper.setMaximum(0.2)
        self.constant_bound_upper.setSingleStep(0.01)
        self.constant_bound_upper.setProperty("value", 0.01)
        self.constant_bound_upper.setObjectName("constant_bound_upper")
        self.gridLayout_2.addWidget(self.constant_bound_upper, 2, 4, 1, 1)
        self.initial_bounds_label = QtWidgets.QLabel(self.layoutWidget)
        self.initial_bounds_label.setObjectName("initial_bounds_label")
        self.gridLayout_2.addWidget(self.initial_bounds_label, 3, 0, 1, 1)
        self.vpp_bound_initial = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.vpp_bound_initial.setMaximum(200.0)
        self.vpp_bound_initial.setSingleStep(0.01)
        self.vpp_bound_initial.setProperty("value", 1.0)
        self.vpp_bound_initial.setObjectName("vpp_bound_initial")
        self.gridLayout_2.addWidget(self.vpp_bound_initial, 3, 1, 1, 1)
        self.decay_time_bound_initial = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.decay_time_bound_initial.setMaximum(100.0)
        self.decay_time_bound_initial.setSingleStep(0.01)
        self.decay_time_bound_initial.setProperty("value", 0.4)
        self.decay_time_bound_initial.setObjectName("decay_time_bound_initial")
        self.gridLayout_2.addWidget(self.decay_time_bound_initial, 3, 2, 1, 1)
        self.vpp_bound_head_lay = QtWidgets.QHBoxLayout()
        self.vpp_bound_head_lay.setObjectName("vpp_bound_head_lay")
        self.vpp_bound_label = QtWidgets.QLabel(self.layoutWidget)
        self.vpp_bound_label.setObjectName("vpp_bound_label")
        self.vpp_bound_head_lay.addWidget(self.vpp_bound_label)
        self.vpp_bound_fixed_check = QtWidgets.QCheckBox(self.layoutWidget)
        self.vpp_bound_fixed_check.setText("")
        self.vpp_bound_fixed_check.setObjectName("vpp_bound_fixed_check")
        self.vpp_bound_head_lay.addWidget(self.vpp_bound_fixed_check)
        self.gridLayout_2.addLayout(self.vpp_bound_head_lay, 0, 1, 1, 1)
        self.fixed_bounds_label = QtWidgets.QLabel(self.layoutWidget)
        self.fixed_bounds_label.setObjectName("fixed_bounds_label")
        self.gridLayout_2.addWidget(self.fixed_bounds_label, 4, 0, 1, 1)
        self.decay_time_bound_fixed = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.decay_time_bound_fixed.setMaximum(100.0)
        self.decay_time_bound_fixed.setSingleStep(0.1)
        self.decay_time_bound_fixed.setProperty("value", 0.1)
        self.decay_time_bound_fixed.setObjectName("decay_time_bound_fixed")
        self.gridLayout_2.addWidget(self.decay_time_bound_fixed, 4, 2, 1, 1)
        self.constant_bound_fixed = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.constant_bound_fixed.setMinimum(0.0)
        self.constant_bound_fixed.setMaximum(0.2)
        self.constant_bound_fixed.setSingleStep(0.01)
        self.constant_bound_fixed.setProperty("value", 0.0)
        self.constant_bound_fixed.setObjectName("constant_bound_fixed")
        self.gridLayout_2.addWidget(self.constant_bound_fixed, 4, 4, 1, 1)
        self.vpp_bound_fixed = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.vpp_bound_fixed.setMinimum(0.01)
        self.vpp_bound_fixed.setMaximum(200.0)
        self.vpp_bound_fixed.setProperty("value", 0.5)
        self.vpp_bound_fixed.setObjectName("vpp_bound_fixed")
        self.gridLayout_2.addWidget(self.vpp_bound_fixed, 4, 1, 1, 1)
        self.frequency_bound_fixed = QtWidgets.QDoubleSpinBox(self.layoutWidget)
        self.frequency_bound_fixed.setMaximum(1000.0)
        self.frequency_bound_fixed.setProperty("value", 275.0)
        self.frequency_bound_fixed.setObjectName("frequency_bound_fixed")
        self.gridLayout_2.addWidget(self.frequency_bound_fixed, 4, 3, 1, 1)
        self.decay_time_head_lay = QtWidgets.QHBoxLayout()
        self.decay_time_head_lay.setObjectName("decay_time_head_lay")
        self.decay_time_bound_label = QtWidgets.QLabel(self.layoutWidget)
        self.decay_time_bound_label.setObjectName("decay_time_bound_label")
        self.decay_time_head_lay.addWidget(self.decay_time_bound_label)
        self.decay_time_bound_fixed_check = QtWidgets.QCheckBox(self.layoutWidget)
        self.decay_time_bound_fixed_check.setText("")
        self.decay_time_bound_fixed_check.setObjectName("decay_time_bound_fixed_check")
        self.decay_time_head_lay.addWidget(self.decay_time_bound_fixed_check)
        self.gridLayout_2.addLayout(self.decay_time_head_lay, 0, 2, 1, 1)
        self.frequency_head_lay = QtWidgets.QHBoxLayout()
        self.frequency_head_lay.setObjectName("frequency_head_lay")
        self.frequency_bound_label = QtWidgets.QLabel(self.layoutWidget)
        self.frequency_bound_label.setObjectName("frequency_bound_label")
        self.frequency_head_lay.addWidget(self.frequency_bound_label)
        self.frequency_bound_fixed_check = QtWidgets.QCheckBox(self.layoutWidget)
        self.frequency_bound_fixed_check.setText("")
        self.frequency_bound_fixed_check.setObjectName("frequency_bound_fixed_check")
        self.frequency_head_lay.addWidget(self.frequency_bound_fixed_check)
        self.gridLayout_2.addLayout(self.frequency_head_lay, 0, 3, 1, 1)
        self.constant_head_lay = QtWidgets.QHBoxLayout()
        self.constant_head_lay.setObjectName("constant_head_lay")
        self.constant_bound_label = QtWidgets.QLabel(self.layoutWidget)
        self.constant_bound_label.setObjectName("constant_bound_label")
        self.constant_head_lay.addWidget(self.constant_bound_label)
        self.constant_bound_fixed_check = QtWidgets.QCheckBox(self.layoutWidget)
        self.constant_bound_fixed_check.setText("")
        self.constant_bound_fixed_check.setObjectName("constant_bound_fixed_check")
        self.constant_head_lay.addWidget(self.constant_bound_fixed_check)
        self.gridLayout_2.addLayout(self.constant_head_lay, 0, 4, 1, 1)
        self.bounds_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.bounds_checkbox.setGeometry(QtCore.QRect(10, 160, 81, 17))
        self.bounds_checkbox.setChecked(True)
        self.bounds_checkbox.setObjectName("bounds_checkbox")
        self.initial_checkbox = QtWidgets.QCheckBox(self.groupBox)
        self.initial_checkbox.setEnabled(True)
        self.initial_checkbox.setGeometry(QtCore.QRect(100, 160, 91, 17))
        self.initial_checkbox.setChecked(False)
        self.initial_checkbox.setObjectName("initial_checkbox")
        self.decay_time_bound_initial_2 = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.decay_time_bound_initial_2.setGeometry(QtCore.QRect(520, 110, 78, 20))
        self.decay_time_bound_initial_2.setMaximum(1.0)
        self.decay_time_bound_initial_2.setSingleStep(0.01)
        self.decay_time_bound_initial_2.setProperty("value", 0.4)
        self.decay_time_bound_initial_2.setObjectName("decay_time_bound_initial_2")
        self.constant_bound_initial_2 = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.constant_bound_initial_2.setGeometry(QtCore.QRect(690, 110, 79, 20))
        self.constant_bound_initial_2.setMaximum(0.2)
        self.constant_bound_initial_2.setSingleStep(0.01)
        self.constant_bound_initial_2.setObjectName("constant_bound_initial_2")
        self.frequency_bound_initial_2 = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.frequency_bound_initial_2.setGeometry(QtCore.QRect(604, 110, 80, 20))
        self.frequency_bound_initial_2.setMaximum(1000.0)
        self.frequency_bound_initial_2.setSingleStep(1.0)
        self.frequency_bound_initial_2.setProperty("value", 300.0)
        self.frequency_bound_initial_2.setObjectName("frequency_bound_initial_2")
        self.vpp_bound_initial_2 = QtWidgets.QDoubleSpinBox(self.groupBox)
        self.vpp_bound_initial_2.setGeometry(QtCore.QRect(435, 110, 79, 20))
        self.vpp_bound_initial_2.setMaximum(200.0)
        self.vpp_bound_initial_2.setSingleStep(0.01)
        self.vpp_bound_initial_2.setProperty("value", 0.1)
        self.vpp_bound_initial_2.setObjectName("vpp_bound_initial_2")

        self.retranslateUi(Dialog)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.groupBox.setTitle(_translate("Dialog", "Fitting Bounds"))
        self.vpp_bound_upper.setSuffix(_translate("Dialog", " mV"))
        self.upper_bounds_label.setText(_translate("Dialog", "Upper"))
        self.lower_bounds_label.setText(_translate("Dialog", "Lower"))
        self.vpp_bound_lower.setSuffix(_translate("Dialog", " mV"))
        self.frequency_bound_lower.setSuffix(_translate("Dialog", " Hz"))
        self.decay_time_bound_upper.setSuffix(_translate("Dialog", " ms"))
        self.decay_time_bound_lower.setSuffix(_translate("Dialog", " ms"))
        self.frequency_bound_initial.setSuffix(_translate("Dialog", " Hz"))
        self.frequency_bound_upper.setSuffix(_translate("Dialog", " Hz"))
        self.initial_bounds_label.setText(_translate("Dialog", "Initial"))
        self.vpp_bound_initial.setSuffix(_translate("Dialog", " mV"))
        self.decay_time_bound_initial.setSuffix(_translate("Dialog", " ms"))
        self.vpp_bound_label.setText(_translate("Dialog", "Vpp"))
        self.fixed_bounds_label.setText(_translate("Dialog", "Fixed"))
        self.decay_time_bound_fixed.setSuffix(_translate("Dialog", " ms"))
        self.vpp_bound_fixed.setSuffix(_translate("Dialog", " mV"))
        self.frequency_bound_fixed.setSuffix(_translate("Dialog", " Hz"))
        self.decay_time_bound_label.setText(_translate("Dialog", "t_0"))
        self.frequency_bound_label.setText(_translate("Dialog", "f"))
        self.constant_bound_label.setText(_translate("Dialog", "c"))
        self.bounds_checkbox.setText(_translate("Dialog", "Use bounds"))
        self.initial_checkbox.setText(_translate("Dialog", "Use initial val"))
        self.decay_time_bound_initial_2.setSuffix(_translate("Dialog", " s"))
        self.frequency_bound_initial_2.setSuffix(_translate("Dialog", " Hz"))
        self.vpp_bound_initial_2.setSuffix(_translate("Dialog", " mV"))


class ParamsDialog(QtWidgets.QDialog, Ui_Dialog):
    def __init__(self,parent):
        QtWidgets.QDialog.__init__(self,parent=parent)
        self.parent=parent
        super(ParamsDialog,self).setupUi(self)

        self.vpp_bound_fixed_check.setCheckState(parent.vpp_bound_fixed_check.checkState())
        self.decay_time_bound_fixed_check.setCheckState(parent.decay_time_bound_fixed_check.checkState())
        self.frequency_bound_fixed_check.setCheckState(parent.frequency_bound_fixed_check.checkState())
        self.constant_bound_fixed_check.setCheckState(parent.frequency_bound_fixed_check.checkState())

        self.vpp_bound_fixed.setValue(parent.vpp_bound_fixed.value())
        self.decay_time_bound_fixed.setValue(parent.decay_time_bound_fixed.value())
        self.frequency_bound_fixed.setValue(parent.frequency_bound_fixed.value())
        self.constant_bound_fixed.setValue(parent.constant_bound_fixed.value())

        self.bounds_checkbox.setCheckState(parent.bounds_checkbox.checkState())
        
        self.vpp_bound_lower.setValue(parent.vpp_bound_lower.value())
        self.decay_time_bound_lower.setValue(parent.decay_time_bound_lower.value())
        self.frequency_bound_lower.setValue(parent.frequency_bound_lower.value())
        self.constant_bound_lower.setValue(parent.constant_bound_lower.value())

        self.vpp_bound_upper.setValue(parent.vpp_bound_upper.value())
        self.decay_time_bound_upper.setValue(parent.decay_time_bound_upper.value())
        self.frequency_bound_upper.setValue(parent.frequency_bound_upper.value())
        self.constant_bound_upper.setValue(parent.constant_bound_upper.value())

        self.vpp_bound_initial.setValue(parent.vpp_bound_initial.value())
        self.decay_time_bound_initial.setValue(parent.decay_time_bound_initial.value())
        self.frequency_bound_initial.setValue(parent.frequency_bound_initial.value())
        self.constant_bound_initial.setValue(parent.constant_bound_initial.value())
    
    def accept(self):
        
        self.parent.vpp_bound_fixed_check.setCheckState(self.vpp_bound_fixed_check.checkState())
        self.parent.decay_time_bound_fixed_check.setCheckState(self.decay_time_bound_fixed_check.checkState())
        self.parent.frequency_bound_fixed_check.setCheckState(self.frequency_bound_fixed_check.checkState())
        self.parent.constant_bound_fixed_check.setCheckState(self.constant_bound_fixed_check.checkState())

        self.parent.vpp_bound_fixed.setValue(self.vpp_bound_fixed.value())
        self.parent.decay_time_bound_fixed.setValue(self.decay_time_bound_fixed.value())
        self.parent.frequency_bound_fixed.setValue(self.frequency_bound_fixed.value())
        self.parent.constant_bound_fixed.setValue(self.constant_bound_fixed.value())

        self.parent.bounds_checkbox.setCheckState(self.bounds_checkbox.checkState())

        self.parent.vpp_bound_lower.setValue(self.vpp_bound_lower.value())
        self.parent.decay_time_bound_lower.setValue(self.decay_time_bound_lower.value())
        self.parent.frequency_bound_lower.setValue(self.frequency_bound_lower.value())
        self.parent.constant_bound_lower.setValue(self.constant_bound_lower.value())

        self.parent.vpp_bound_upper.setValue(self.vpp_bound_upper.value())
        self.parent.decay_time_bound_upper.setValue(self.decay_time_bound_upper.value())
        self.parent.frequency_bound_upper.setValue(self.frequency_bound_upper.value())
        self.parent.constant_bound_upper.setValue(self.constant_bound_upper.value())

        self.parent.vpp_bound_initial.setValue(self.vpp_bound_initial.value())
        self.parent.decay_time_bound_initial.setValue(self.decay_time_bound_initial.value())
        self.parent.frequency_bound_initial.setValue(self.frequency_bound_initial.value())
        self.parent.constant_bound_initial.setValue(self.constant_bound_initial.value())
        super(ParamsDialog,self).accept()