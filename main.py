import sys
import librosa
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import QUrl
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QFileDialog, QSlider
from scipy.io import wavfile
from scipy.signal.windows import boxcar, hanning, hamming
from pop import Ui_Form
from pyqtgraph import PlotWidget
from NewWindow import UIMainWindow

class ApplicationWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ApplicationWindow, self).__init__()
        uic.loadUi('ui.ui', self)
        pg.setConfigOptions(antialias=True)

        self.plots = []
        self.datalines = []

        self.samples = [np.array([]), np.array([]), np.array([])]  # origin, test1, test2
        self.time = 0

        self.fft_samples = [0, 0, 0]
        self.freqs_ = 0

        self.gain_values = np.zeros(20).reshape((2, 10))

        self.pens = [pg.mkPen('r'), pg.mkPen('w'), pg.mkPen(
            'y'), pg.mkPen('g'), pg.mkPen('m'), pg.mkPen('w')]

        for i in range(3):
            plot1 = self.gView.addPlot(title="Time Domain")
            plot2 = self.gView.addPlot(title="Frequency Domain")
            self.gView.nextRow()
            self.plots.append(plot1)
            self.plots.append(plot2)

        for i in range(6):
            self.datalines.append(self.plots[i].plot(pen=self.pens[i]))

        self.linear_regions = []
        self.NUM_BANDS = 10
        self.line_position = 0
        # origin , test1, test2
        self.mediaPlayers = [QMediaPlayer(), QMediaPlayer(), QMediaPlayer()]

        self.timer = QtCore.QTimer()  # for regions on graph
        self.timer.setInterval(180)
        self.timer.timeout.connect(self.removeRegion)

        self.timer1 = QtCore.QTimer()  # for control the playing of audio
        self.timer1.setInterval(100)
        self.timer1.timeout.connect(self.update_postion)

        self.comb_box_current_index = [0,0]
        # UI
        self.freq_bands = [
            self.freq1, self.freq2, self.freq3, self.freq4, self.freq5,
            self.freq6, self.freq7, self.freq8, self.freq9, self.freq10
        ]
        self.gains = [
            self.gain1, self.gain2, self.gain3, self.gain4, self.gain5,
            self.gain6, self.gain7, self.gain8, self.gain9, self.gain10
        ]
        self.sliders = [
            self.slider1, self.slider2, self.slider3, self.slider4, self.slider5,
            self.slider6, self.slider7, self.slider8, self.slider9, self.slider10
        ]

        self.tests = [self.test1, self.test2]
        self.test_is_modified = [0, 0]

        self.playline = self.playGraph.plot()
        self.pen = pg.mkPen(color='r', width=3)
        self.playGraph.setYRange(min=-2, max=2)
        self.playGraph.showAxis('bottom', False)
        self.playGraph.plotItem.autoBtn = None
        self.playGraph.showAxis('left', False)

        self.min_gain = -30  # dB
        self.max_gain = 30  # dB
        self.disable_some_uis()

        # connections

        self.add_file_btn.clicked.connect(self.add_file)
        self.play_btn.clicked.connect(self.play_audio)
        self.pause_btn.clicked.connect(self.pause_audio)
        self.orgin_radio.clicked.connect(self.stop_play)
        self.test_radio.clicked.connect(self.stop_play)
        self.show_dif_btn.clicked.connect(self.show_popup)
        self.pushButton.clicked.connect(self.show_popup2)
        self.save_btn.clicked.connect(self.save)
        self.reset_btn.clicked.connect(self.reset_sliders)
        self.actionNew_Window.triggered.connect(lambda: self.new_win())

        for i in range(2):
            self.connect_tests(i)
        for i in range(self.NUM_BANDS):
            self.connect_sliders(i)

        self.is_playing = False



    def connect_sliders(self, index):
        self.sliders[index].valueChanged.connect(
            lambda: self.modify_gain(index))

    def connect_tests(self, i):
        self.tests[i].clicked.connect(lambda: self.get_gains(i))



    def add_file(self):
        filename = QFileDialog(self).getOpenFileName()
        path = filename[0]
        if path != '':
            if self.samples[0].size != 0:
                self.reset()

            #---------------- Time domain -----------------
            self.samples[0], self.sampling_freq = librosa.load(path, sr=None)     #Samples are the points we'll draw
            if self.samples[0].size % 2 == 0:
                np.append(self.samples[0], 0)
            self.freq_max = self.sampling_freq / 2
            self.time = np.arange(0, self.samples[0].size)

            #--------------Freq domain-----------------
            self.num_samples = 2 ** int(np.ceil(np.log2(self.samples[0].size)))
            self.fft_samples[0] = np.fft.rfft(
                self.samples[0], self.num_samples)
            self.freqs = np.linspace(
                0, self.freq_max, self.fft_samples[0].size)         #diff from arrang() that define step

            for i in range(1, 3):
                self.samples[i] = self.samples[0].copy()
                self.fft_samples[i] = self.fft_samples[0].copy()

            '''
            Unable Everything till we add a file
            '''
            self.get_bands()
            self.get_windows()
            self.enable_some_uis()
            self.set_bands_gains_sliders()
            self.draw_origin()

            self.test1.setChecked(True)
            self.orgin_radio.setChecked(True)
            self.test_radio.setDisabled(True)

            #--------------playing a graph-----------------
            self.vLine = pg.InfiniteLine(angle=90, movable=True, pen=self.pen)
            self.vLine.sigPositionChangeFinished.connect(self.update_p)
            self.playGraph.addItem(self.vLine)
            self.playGraph.setLimits(
                xMin=0, xMax=self.samples[0].size + 30, yMin=-1, yMax=1)            #Sample points number adding 30 excess
            self.playGraph.setXRange(min=0, max=self.samples[0].size + 30)
            self.playline.setData(self.time, self.samples[0])
            self.playGraph.autoRange()
            self.mediaPlayers[0].setMedia(
                QMediaContent(QUrl.fromLocalFile(path)))


    def draw_origin(self):
        for i in range(6):
            if i % 2 == 0:
                self.datalines[i].setData(self.time, self.samples[0])
                self.plots[i].autoRange()
            else:
                self.datalines[i].setData(
                    self.freqs, (2 / self.num_samples) * np.abs(self.fft_samples[0]))
                self.plots[i].autoRange()


    def get_bands(self):
        self.bands = np.linspace((self.freq_max / (2 * self.NUM_BANDS)),
                                 self.freq_max - (self.freq_max / (2 * self.NUM_BANDS)), self.NUM_BANDS)

        # rectangle window
        self.band_length = int(np.ceil((self.fft_samples[0].size / self.NUM_BANDS)))
        self.last_band_length = int(self.fft_samples[0].size -
                               (self.NUM_BANDS - 1) * self.band_length)
        self.rc_lengths = [self.band_length, self.last_band_length]

        self.band_rc_bounds = [(i * self.band_length, (i + 1) * self.band_length)
                               for i in range(0, self.NUM_BANDS - 1)]

        last_band_bound = (
            self.band_rc_bounds[self.NUM_BANDS - 2][1], self.fft_samples[0].size)
        self.band_rc_bounds.append(last_band_bound)


        # hanning window
        self.hn_length = int(2 * (self.band_length - 1) + 1)
        # indecies of hanning window of the first and last bands
        hn_index = [(int(np.ceil(self.hn_length / 2 - self.band_length / 2)), self.hn_length),
                    (0, int(np.ceil(self.hn_length / 2 + self.last_band_length / 2)))]

        band_hn_bounds = self.get_h_band_bounds(self.band_length, self.hn_length)


        # hamming window
        self.hm_length = int(
            (2 * np.pi * (self.band_length/2 - 1) / 1.483729844447501) + 1)
        hm_index = [(int(np.ceil(self.hm_length / 2 - self.band_length / 2)), self.hm_length),
                    (0, int(np.ceil(self.hm_length / 2 + self.last_band_length / 2)))]

        band_hm_bounds = self.get_h_band_bounds(self.band_length, self.hm_length)

        self.band_bounds = [self.band_rc_bounds,
                            band_hn_bounds, band_hm_bounds]
        self.h_index = [hn_index, hm_index]
        self.band_lengths = [self.band_length,self.last_band_length]
        self.h_length =[self.hn_length, self.hm_length]



    def get_h_band_bounds(self, band_length, h_length):
        band_h_bounds = [(0, int(np.ceil(h_length / 2 + band_length / 2)))]
        for i in range(1, self.NUM_BANDS - 1):
            index1 = self.band_rc_bounds[i][0]
            index2 = self.band_rc_bounds[i][1]
            mid_index = (index1 + index2) // 2
            bound1 = mid_index - h_length // 2 - 1
            bound2 = mid_index + h_length // 2
            if bound1 < 0:
                bound1 = 0
            if bound2 > self.fft_samples[0].size:
                bound2 = self.fft_samples[0].size
            band_h_bounds.append((bound1, bound2))
        band_h_bounds.append((self.fft_samples[0].size - h_length //
                              2 - band_length // 2 - 1, self.fft_samples[0].size))
        return band_h_bounds


    def get_windows(self):
        rectangle_windows = [
            boxcar(self.rc_lengths[0]), boxcar(self.rc_lengths[1])]
        hanning_windows = hanning(self.hn_length)
        hamming_windows = hamming(self.hm_length)
        self.windows = [rectangle_windows, hanning_windows, hamming_windows]


    def get_test(self):
        if self.test1.isChecked():
            return 1
        else:
            return 2


    def set_bands_gains_sliders(self):
        for i in range(self.NUM_BANDS):
            self.freq_bands[i].setText(str(self.bands[i]))
            self.freq_bands[i].adjustSize()
            self.gains[i].setText('0')
            self.sliders[i].setMinimum(self.min_gain)
            self.sliders[i].setMaximum(self.max_gain)
            self.sliders[i].setValue(0)
            self.sliders[i].setTickInterval(1)
            self.sliders[i].setTickPosition(QSlider.TicksLeft)

# ----------------For Disabling Ui's when they're not in use

    def enable_some_uis(self):
        for i in range(self.NUM_BANDS):
            self.sliders[i].setEnabled(True)
            self.freq_bands[i].setEnabled(True)
            self.gains[i].setEnabled(True)

    def disable_some_uis(self):
        for i in range(self.NUM_BANDS):
            self.sliders[i].setDisabled(True)
            self.freq_bands[i].setDisabled(True)
            self.gains[i].setDisabled(True)



    def draw(self, graph_num):
        self.datalines[graph_num *
                       2].setData(self.time, self.samples[graph_num])
        self.datalines[graph_num * 2 + 1].setData(
            self.freqs, (2 / self.num_samples) * np.abs(self.fft_samples[graph_num]))
        self.plots[graph_num * 2].autoRange()
        self.plots[graph_num * 2 + 1].autoRange()

    def removeRegion(self):
        for l in self.linear_regions:
            if self.test1.isChecked():  # get current channel
                self.plots[3].removeItem(l)
                self.plots[3].autoRange()
            else:
                self.plots[5].removeItem(l)
                self.plots[5].autoRange()
            self.linear_regions.remove(l)
        if len(self.linear_regions) == 0:
            self.timer.stop()


    def update_gain_values(self, slider_num, val):
        test = self.get_test()
        self.gain_values[test - 1, slider_num] = val
        self.gains[slider_num].setText(str(val))


    def modify_gain(self, slider_num):
        self.test_radio.setEnabled(True)
        if self.test2.isChecked():
            self.test_is_modified[1] = True
        else:
            self.test_is_modified[0] = True
        val = self.sliders[slider_num].value()
        self.update_gain_values(slider_num, val)
        val = 10 ** (val / 20)  # convert dB
        # get window
        window = self.window_comb.currentIndex()
        begin_index = self.band_bounds[window][slider_num][0]
        end_index = self.band_bounds[window][slider_num][1]

        test = self.get_test()

        self.comb_box_current_index[test-1] = window

        if window == 0:
            self.fft_samples[test][begin_index:end_index] = val * self.windows[0][
                slider_num // 9] * self.fft_samples[0][begin_index:end_index]
        else:
            if slider_num != 0 and slider_num != 9:

                if begin_index == 0:
                    w_begin_index = self.windows[window].size - \
                        self.fft_samples[0][begin_index:end_index].size
                    self.fft_samples[test][begin_index:end_index] = val * self.windows[window][
                        w_begin_index:] * self.fft_samples[0][
                        begin_index:end_index]

                elif end_index == self.fft_samples[0].size:
                    w_end_interval = self.windows[window].size - \
                        self.fft_samples[0][begin_index:end_index].size
                    self.fft_samples[test][begin_index:end_index] = val * self.windows[window][:self.windows[
                        window].size - w_end_interval] *self.fft_samples[0][begin_index:end_index]
                else:
                    ##############################################

                    begin = self.band_rc_bounds[slider_num][0]
                    end = self.band_rc_bounds[slider_num][1]

                    # print(self.h_length[window-1])
                    val1 = self.sliders[slider_num -1].value()
                    val1 = 10 **(val1/20)
                    val2 = self.sliders[slider_num + 1].value()
                    val2 = 10 ** (val2 / 20)

                    # getting values at ends before changed
                    val1_array = val1*self.windows[window][0:int(self.h_length[window-1] / 2 - self.band_lengths[slider_num // 9] / 2)]*self.fft_samples[0][begin-int(self.h_length[window-1] / 2 - self.band_lengths[slider_num // 9] / 2): begin]
                    val2_array = val2*self.windows[window][self.h_length[window-1] - int(self.h_length[window-1] / 2 - self.band_lengths[slider_num // 9] / 2):]* self.fft_samples[0][end: int(self.h_length[window-1] / 2 - self.band_lengths[slider_num // 9] / 2) + end]

                    # apply window
                    self.fft_samples[test][begin_index:end_index] = val * self.windows[window] * self.fft_samples[0][begin_index:end_index]

                    # sum values at ends
                    self.fft_samples[test][begin - int(self.h_length[window-1] / 2 - self.band_lengths[slider_num // 9] / 2):begin] += val1_array
                    self.fft_samples[test][end:int(self.h_length[window-1] / 2 - self.band_lengths[slider_num // 9] / 2)+end] += val2_array

                    ##################################################

            else:
                h_index = self.h_index[window // 2]
                w_begin_index = h_index[slider_num // 9][0]
                w_end_index = h_index[slider_num // 9][1]
                self.fft_samples[test][begin_index:end_index] = val * self.windows[window][
                    w_begin_index:w_end_index] * self.fft_samples[0][
                    begin_index:end_index]



        begin_band = int(self.bands[slider_num] - self.freq_max/20)
        end_band = int(self.bands[slider_num] + self.freq_max/20)


        self.linear_regions.append(pg.LinearRegionItem(
            [begin_band, end_band], movable=False))
        self.plots[test * 2 + 1].addItem(self.linear_regions[-1])

        self.samples[test] = np.fft.irfft(self.fft_samples[test])[
            :self.samples[0].size]
        self.draw(test)
        self.timer.start()

        wavfile.write('.test{}.wav'.format(test),
                      self.sampling_freq, self.samples[test])

        self.mediaPlayers[test].setMedia(QMediaContent(
            QUrl.fromLocalFile(".test{}.wav".format(test))))



        if self.test_radio.isChecked():  # this is for continous playing during modifying
            if self.is_playing == True:
                self.mediaPlayers[test].setPosition(
                    (1 / self.sampling_freq) * self.vLine.getPos()[0] * 10 ** 3)
                self.mediaPlayers[test].play()




    def get_gains(self, test):
        for i in range(self.NUM_BANDS):
            self.gains[i].setText(str(self.gain_values[test, i]))
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(self.gain_values[test, i])
            self.sliders[i].blockSignals(False)
        self.window_comb.setCurrentIndex(self.comb_box_current_index[test])

    def new_win(self):
        self.window = UIMainWindow()
        self.window.show()


    def play_audio(self):
        position = self.vLine.getPos()[0]
        test = self.get_test()
        if self.orgin_radio.isChecked():
            # end of play and user play again
            if position == self.samples[0].size - 1:
                self.mediaPlayers[0].setPosition(0)
            else:
                self.mediaPlayers[0].setPosition(
                    (1 / self.sampling_freq) * position * 10 ** 3)
            self.mediaPlayers[0].play()
        else:
            if position == self.samples[test].size - 1:
                self.mediaPlayers[test].setPosition(0)
            else:
                self.mediaPlayers[test].setPosition(
                    (1 / self.sampling_freq) * position * 10 ** 3)
            self.mediaPlayers[test].play()

        self.is_playing = True
        self.timer1.start()



    def update_postion(self):
        self.line_position += 0.1 * self.sampling_freq
        if self.line_position > self.samples[0].size:
            self.vLine.setPos(self.samples[0].size - 1)
            self.line_position = 0
            self.timer1.stop()
        else:
            self.vLine.setPos(self.line_position)


    def update_p(self):
        self.line_position = self.vLine.getPos()[0]


    def pause_audio(self):
        test = self.get_test()
        if self.orgin_radio.isChecked():
            self.mediaPlayers[0].pause()
        else:
            self.mediaPlayers[test].pause()
        self.timer1.stop()
        self.is_playing = False


    def stop_play(self):
        self.line_position = 0
        self.vLine.setPos(0)
        self.timer1.stop()
        test = self.get_test()
        self.mediaPlayers[0].stop()

        self.mediaPlayers[test].stop()
        self.is_playing = False



    def show_popup2(self):
        self.Form = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.Form)
        plots = []
        datalines = []
        no_plots = 1
        if self.test_is_modified[1] == True:
            no_plots = 2

        for i in range(no_plots):
            plots.append(self.ui.pop_graph.addPlot(
                title="Spectrogram".format(i + 1)))
        for i in range(no_plots):
            datalines.append(plots[i].plot(pen=self.pens[i]))
        self.Form.show()


    def show_popup(self):
        self.Form = QtWidgets.QWidget()
        self.ui = Ui_Form()
        self.ui.setupUi(self.Form)
        plots = []
        datalines = []
        no_plots = 1
        if self.test_is_modified[1] == True:
            no_plots = 2

        for i in range(no_plots):
            plots.append(self.ui.pop_graph.addPlot(
                title="Time Domain Difference test{}".format(i + 1)))
            plots.append(self.ui.pop_graph.addPlot(
                title="Freq Domain Difference test{}".format(i + 1)))
            self.ui.pop_graph.nextRow()
        for i in range(no_plots * 2):
            datalines.append(plots[i].plot(pen=self.pens[i]))

        # time domains
        for i in range(0, no_plots * 2, 2):
            datalines[i].setData(
                self.time, (self.samples[0] - self.samples[(i + 2) // 2]))
        # freq domain
        for i in range(1, no_plots * 2, 2):
            datalines[i].setData(self.freqs, abs(
                self.fft_samples[0] - self.fft_samples[(i + 2) // 2]))
        self.Form.show()


    def reset(self):
        self.vLine.setPos(0)
        self.timer1.stop()
        self.line_position = 0
        for i in range(3):
            self.mediaPlayers[i].stop()
        self.test_radio.setDisabled(True)
        self.test1.setChecked(True)
        self.test_is_modified = [0, 0]


    def save(self):
        test = self.get_test()
        name = QtGui.QFileDialog.getSaveFileName(self, 'Save file', 'test.wav')
        path = name[0]
        if path != '':
            wavfile.write(path, self.sampling_freq, self.samples[test])

    def reset_sliders(self):
        test = self.get_test()
        for i in range(self.NUM_BANDS):
            self.gains[i].setText('0')
            self.gain_values[test -1,i] = 0
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(0)
            self.sliders[i].blockSignals(False)
        self.samples[test] = self.samples[0].copy()
        self.fft_samples[test] = self.fft_samples[0].copy()
        self.draw(test)
        wavfile.write('.test{}.wav'.format(test),
                      self.sampling_freq, self.samples[test])

        self.mediaPlayers[test].setMedia(QMediaContent(
            QUrl.fromLocalFile('.test{}.wav'.format(test))))
        self.timer1.stop()
        self.vLine.setPos(0)
        self.line_position = 0


def main():
    app = QtWidgets.QApplication(sys.argv)
    application = ApplicationWindow()
    application.show()
    app.exec_()


if __name__ == "__main__":
    main()

# actionNew_Window