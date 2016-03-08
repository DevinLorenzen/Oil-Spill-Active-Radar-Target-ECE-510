#!/usr/bin/env pythonw

# radar analysis of audio data


import pyaudio
import sys
import wavio
import scipy.signal as sig
import threading
import time
import numpy as np
import scipy.signal as sig
import wx
import matplotlib
import argparse
from itertools import groupby
from operator import itemgetter
matplotlib.use('WXAgg')
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx
from matplotlib.figure import Figure


#### PARAMETERS OF SPECTROGRAM
# sample rate of sound card
TS = 8000

# number of samples to get from sound card at once
CHUNK = 1024
# How big the gui should be 
SIZE = 512
NOVL = 2*SIZE*15/16
# how many windows of data to keep for the psd
NUM_WINDOWS=8

# how many sigma out the signal must be before it is considered significant
NUM_SIGMA=100

# DO NOT MODIFY
Fs=1/float(TS)

class SpectroPanel(wx.Panel):
    """Panel that holds a matplot thingy"""
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.graph = np.zeros((SIZE,SIZE),np.float32)
        data = np.uint8(255*cm.hot(self.graph)[:,:,0:3])
        self.bmp = wx.BitmapFromBuffer(SIZE, SIZE, data)
        # create a buffer that can be used to store historical samples
        self.sample_buffer=np.zeros(NUM_WINDOWS*CHUNK,np.float32)
        self.sample_pos = 0 # start a first sample location
        self.sample_count = -1
        self.graph_pos = 0 # which time line we are at
        self.in_setup = True


        # called when a chunk of data is read from the microphone
    def callback(self,in_data, frame_count, time_info, status):
        start = self.sample_pos*CHUNK
        stop = (self.sample_pos+1)*CHUNK
        window = np.blackman(2*SIZE)
        # get data from microphone
        data = np.float32(in_data)
        self.sample_buffer[start:stop] = data
        self.sample_count+=1
        self.sample_pos = self.sample_pos+1
        if self.sample_pos == NUM_WINDOWS:
            self.sample_pos = 0

            # if this is the first time through, we record the average power/hz to use for thresholding
            if self.in_setup:
                freq,psd = sig.welch(np.roll(self.sample_buffer,-self.sample_pos*CHUNK),
                                     fs=Fs, nfft=2*SIZE, scaling='spectrum', noverlap=NOVL, window=window)
                self.threshold = np.mean(psd) + NUM_SIGMA*np.std(psd)
            self.in_setup = False
            
        # we need to wait until we have gathered at least NUM_WINDOWS
        # buffers, so we don't divide by zero
        if self.in_setup:
            return (None, pyaudio.paContinue)

        # calculate psd
        freq,psd = sig.welch(np.roll(self.sample_buffer,-self.sample_pos*CHUNK),
                             fs=Fs, nfft=2*SIZE, scaling='spectrum')

        # copy over all the frequency data to that column
        self.graph[:,self.graph_pos] = psd[:-1]


        # label all points greater than the threshold as marked
        indices = [i for i,v in enumerate(psd[:-1]) if v > self.threshold]
        self.graph[indices,self.graph_pos] = np.nan
        self.graph_pos = (self.graph_pos + 1) % SIZE

        if len(indices) > 0:
            runs = [map(itemgetter(1), g) for k, g in groupby(enumerate(indices), lambda (i, x): i-x)]
            output = [str(float(self.sample_count)*CHUNK/TS)]
            output.extend([str(np.mean(r)*TS / SIZE / 2) for r in runs])
            print ",".join(output)
        
        # rotate data so we get a nice scrolling spectrogram
        unscaled_data = np.roll(self.graph,-self.graph_pos,1)

        # scale graph to take full example of 8 bit
        # display. Spectrogram floating point power values scaled to be between 0-255
        minval = np.nanmin(unscaled_data)
        minval = 0
        maxval = np.nanmax(unscaled_data) - minval

        scaled_data = (unscaled_data-minval)/maxval

        # create a bitmap of the colored spectrogram
        colored = np.uint8(255*matplotlib.cm.hot(scaled_data)[:,:,0:3])
        # make all thresholded values blue
        colored[np.isnan(scaled_data)] = [0,0,255]

        # update the bmp
        self.bmp = wx.BitmapFromBuffer(SIZE, SIZE, colored)
        self.Refresh()
        return (None, pyaudio.paContinue)

    # update display
    def OnPaint(self, evt):
        dc = wx.PaintDC(self)
        dc.DrawBitmap(self.bmp,0,0)

class WavThread(threading.Thread):
    def __init__(self, event,data,panel):
        threading.Thread.__init__(self)
        self.stopped = event
        self.data = data
        self.panel = panel

    def run(self):
        i = 0
        while not self.stopped.wait(0.1):
            d = self.data[i*CHUNK:(i+1)*CHUNK]

            if len(d) == CHUNK:
                self.panel.callback(d,None,None,None)
            else:
                self.stopped.set()
            i+= 1
# main task        
def main(args):
    # setup gui
    app = wx.App()
    fr = wx.Frame(None, title='RADAR Spectrogram')
    fr.SetSize((SIZE,SIZE))
    panel = SpectroPanel(fr)

    # turn on sound card
    p = pyaudio.PyAudio()
    # stream = p.open(format=pyaudio.paInt16,
    #                 channels=1,
    #                 rate=TS,
    #                 output=False,
    #                 input=True,
    #                 frames_per_buffer=CHUNK,
    #                 stream_callback=panel.callback)

    # start stuff
#    stream.start_stream()

    wav = wavio.read(args.filename)
    data = wav.data
    rate = wav.rate
    if data.shape[1] > 1:
        data = data[:,1]
    else:
        data = data.transpose()

    if rate != TS:
        data = sig.decimate(data,rate/TS,ftype='fir')

    stopFlag = threading.Event()
    thread = WavThread(stopFlag,data,panel)
    thread.start()

    fr.Show()
    app.MainLoop()

#    stream.stop_stream()
#    stream.close()
    stopFlag.set()
    #p.terminate()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='the file to parse')
    main(parser.parse_args())
    
