import os
import sys
'''import math
import time
import h5py #модуль підтримки файлів'''
#from PyQt5.QtWidgets import (QMainWindow, QDesktopWidget, QMessageBox, QWidget, QToolTip,
#    QPushButton, QTextEdit, QCheckBox, QFileDialog, QLineEdit, QLabel, QApplication, qApp, QAction, QVBoxLayout)
from PyQt5 import QtWidgets
'''from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon #UI
import numpy as np  # модуль масивів
from scipy.interpolate import CubicSpline  # кубічна інтерполяція
from scipy.optimize import minimize  # знах екстремумів
import matplotlib as mpl  # модуль відображення графіків
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque  # імпортує клас черги collections.deque'''

import design
from Water import visualize

'''
class SecondWindow(QWidget):
    def __init__(self, parent=None):
        # Передаём ссылку на родительский элемент и чтобы виджет
        # отображался как самостоятельное окно указываем тип окна
        super().__init__(parent, Qt.Window)
        self.build()

    def build(self):
        self.mainLayout = QVBoxLayout()
        self.textEdit = QLineEdit(self)
        self.textEdit.setReadOnly(True)
        self.textEdit.setText("Всі права захищено")
        self.textEdit.move(0, 0)
        self.textEdit.resize(200, 300)
        self.textEdit = QTextEdit()
        self.setGeometry(0, 0, 200, 300)
        self.setWindowTitle('Довідка')
        self.setLayout(self.mainLayout)

class MainWindow(QWidget):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.secondWin = None
        self.initUI()


    def initUI(self):
        # вспливаючі підказки
        QToolTip.setFont(QFont('SansSerif', 10))

        #іконка
        self.setWindowTitle('Icon')
        self.setWindowIcon(QIcon('pattern.png'))

        btn = QPushButton('Старт', self)
        btn.setToolTip('Почати виконання програми')
        btn.resize(btn.sizeHint())
        btn.move(50, 500)
        #btn.clicked.connect(modeling())

        btn_help = QPushButton('Довідка', self)
        btn_help.setToolTip('Інформація про програму')
        btn_help.resize(btn.sizeHint())
        btn_help.move(210, 500)
        btn_help.clicked.connect(self.openWin)

        btn_help = QPushButton('Вихід', self)
        btn_help.setToolTip('Вийти на робочий стіл')
        btn_help.resize(btn.sizeHint())
        btn_help.move(370, 500)

        hi_label = QLabel("Вітаємо! Введіть ваші заміри у поле знизу, або прикріпіть файл:", self)
        hi_label.move(50, 20)

        cb = QCheckBox('Зберегти анімацію', self)
        cb.move(50, 450)
        cb.toggle()
        cb.stateChanged.connect(self.changeTitle)

        btn_file = QPushButton('Файл...', self)
        btn_file.move(50, 50)
        btn_file.setToolTip('Прикріпляйте файл розширення <b>.txt</b>')
        btn_file.clicked.connect(self.showDialog)

        exitAction = QAction(QIcon('exit.png'), '&Вихід', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

        startAction = QAction('&Старт', self)
        exitAction.setShortcut('Ctrl+Q')
        exitAction.setStatusTip('Exit application')
        exitAction.triggered.connect(qApp.quit)

       
        self.textEdit = QLineEdit(self)
        self.textEdit.move(50, 100)
        self.textEdit.resize(400,300)
        self.textEdit.setToolTip('Числа записуються через кому (,), знаки після коми відокремлюються крапкою (.). Приклад: 2.0, 3.7')
        self.textEdit = QTextEdit()
        #self.setCentralWidget(self.textEdit)


        self.setGeometry(100, 100, 500, 600)
        self.setWindowTitle('Моделювання затоплення поверхні')

        self.show()

    def openWin(self):
        if not self.secondWin:
             self.secondWin = SecondWindow(self)
        self.secondWin.show()

        # закриття програми
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Вихід', "Ви дійсно бажаєте вийти?", QMessageBox.Yes | QMessageBox.No,  QMessageBox.No)
        if reply == QMessageBox.Yes:
            event.accept()
        else:
             event.ignore()

    #центрування вікна
    def center(self):
        qr = self.frameGeometry()
        cp = QDesktopWidget().availableGeometry().center()
        qr.moveCenter(cp)
        self.move(qr.topLeft())

    #збереження анімації
    #TODO збереження анымації по стану чекбокса
    def changeTitle(self, state):
        if state == Qt.Checked:
            self.setWindowTitle('QCheckBox')
        else:
            self.setWindowTitle('QCheckBox')

    def showDialog(self):
        fname = QFileDialog.getOpenFileName(self, 'Open file', '/home')[0]
        f = open(fname, 'r')

        with f:
            data = f.read()
            self.textEdit.setText(data)



def main():
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

#відображ. UI
main()

'''

class ExampleApp(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        # для доступа до змінних, методів
        # в файле design.py
        super().__init__()
        self.setupUi(self)  # ініціалізація дизайна


def main():
    #V_f_all = []
    app = QtWidgets.QApplication(sys.argv)  # новий екземпляр QApplication
    window = ExampleApp()  # ств об'єкт класу ExampleApp

    def start_button_clicked():
        #global V_f_all
        set_intense = float(window.lineEdit.text())

        k = 0
        if window.radioButton_2.isChecked():  # Пісок
            k = 1
        elif window.radioButton_3.isChecked():  # Супісок
            k = 0.1
        elif window.radioButton_4.isChecked():  # Суглинок
            k = 0.01
        elif window.radioButton.isChecked():  # Глина
            k = 0.002
        else:
            raise Exception("unknown material")

        print(window.radioButton_2.text(), window.radioButton_2.isChecked())
        print(set_intense)

        #V_f_all = modeling(set_intense, k)
        visualize(set_intense, k)

        # print("Done! V filter: ", V_f_all)
        window.lineEdit.text("Done!")


    window.pushButton_2.clicked.connect(start_button_clicked)

    window.show()  # показуєм вікно
    app.exec_()  # запускаєм додаток


if __name__ == '__main__':  # якщо запускаэм файл напряму, а не імпортуємо
    main()  # то запускаєм ф-ю main()
