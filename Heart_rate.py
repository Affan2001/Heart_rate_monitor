# import required libraries
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget, plot
from scipy.ndimage import gaussian_filter1d


# Import the butter and filtfilt functions in scipy
from scipy.signal import butter, filtfilt

# Import time module for calculating frame rate
import time

# Define a window class, inherited from QMainWindow
class Ui_MainWindow(QMainWindow):
     # initialization method
     def __init__(self):
         super().__init__()
         # Set the title and size of the window
         self.setWindowTitle("Python Program")
         self. resize(800, 600)
         # Create a video capture object, the parameter is 0 means use the default camera
         self.cap = cv2.VideoCapture(0)
         self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
         self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
         # Create a timer for regularly updating video frame and curve data
         self. timer = QTimer()
         # Create a face detector, using the cascade classifier that comes with opencv
         self.face_detector = cv2.CascadeClassifier("C:\Project\heart\haarcascade_frontalface_default.xml")#)
         # Create an empty numpy array to store raw data and heart rate data
         self.raw_data = np.array([])
         self. filtered_data = np. array([])
         self. heart_rate_data = np. array([0])
         # Create a layout object for placing controls
         self.layout = QtWidgets.QVBoxLayout()
         # Create two buttons for starting and ending the video
         self.start_button = QtWidgets.QPushButton("Start Video")
         self.stop_button = QtWidgets.QPushButton("End Video")
         # Add the button to the layout, and set the signal and slot functions of the button
         self.layout.addWidget(self.start_button)
         self.layout.addWidget(self.stop_button)
         self.start_button.clicked.connect(self.start_video)
         self.stop_button.clicked.connect(self.stop_video)
         # Create a label for displaying video frames
         self.video_label = QtWidgets.QLabel()
         # Add the label to the layout, and set the size and alignment of the label
         self.layout.addWidget(self.video_label)
         self. video_label. setFixedSize(640, 480)
         self.video_label.setAlignment(QtCore.Qt.AlignCenter)
         # Create two drawing controls, used to display the original data curve and heart rate curve respectively
         self.raw_data_plot = PlotWidget()
         self. heart_rate_plot = PlotWidget()
         # Add the drawing control to the layout, and set the title and axis label of the drawing control
         self.layout.addWidget(self.raw_data_plot)
         self.layout.addWidget(self.heart_rate_plot)
         self.raw_data_plot.setTitle("Raw data curve")
         self. heart_rate_plot. setTitle("Frequency curve")
         self.raw_data_plot.setLabel("left", "Mean value of green pixels")
         self.heart_rate_plot.setLabel("left", "Frequency magnitude")
         #self.heart_rate_plot.setYRange(0,200)
         # Set the window's central control to a widget and set the layout to the widget's layout
         self. central_widget = QtWidgets. QWidget()
         self.setCentralWidget(self.central_widget)
         self.central_widget.setLayout(self.layout)

         self.data_len_max = 30*10

         # In the initialization method, create a variable to store the time of the previous frame
         self.last_frame_time = 0
         self.frame_rate = 30.0
         self.frame_rate_list = [30.0]

         self.freq_x = [0]
         self.freq_y = [0]

     # Define a slot function to start the video, which is used to start the timer and set the signal and slot function of the timer
     def start_video(self):
         #self.cap.open(0)
         # Start the timer and trigger the timeout signal every 50 milliseconds
         self. timer. start(20)
         # Connect the timeout signal to the update_frame slot function to update the video frame and curve data
         self.timer.timeout.connect(self.update_frame)
    # Define a slot function that ends the video, used to stop the timer and release the video capture object
     def stop_video(self):
         # Stop the timer and disconnect the timeout signal from the update_frame slot function
         self. timer. stop()
         self.timer.timeout.disconnect(self.update_frame)

     # Define a window close event to release the video capture object
     def closeEvent(self, event):
         # Release the video capture object
         self.cap.release()

     def smooth_signal(self, signal):
         # sigma is the standard deviation of the Gaussian kernel
         # it controls the degree of smoothing
         sigma = 3
         # apply the filter to the signal
         smoothed_signal = gaussian_filter1d(signal, sigma)
         # return the smoothed signal
         return smoothed_signal
     def calc_heart_rate(self, frame):
         # Convert the image to grayscale for face detection
         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
         # Use the face detector to detect faces in the image, return a list, each element is a rectangular area
         faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
         # Loop through each detected face
         for (x, y, w, h) in faces:
             # Draw a green rectangle on the original image to represent the face area
             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
             # Capture the image of the face area and convert it to HSV color space
             face = frame[y:y + h, x:x + w]
             hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
             # Calculate the mean value of the green pixels of the face area image as a point of the original data
             green_mean = np.mean(hsv[:, :, 1])
             # Add the original data points to the original data array, and keep the length of the array not exceeding 100
             self.raw_data = np.append(self.raw_data, green_mean)
             self.filtered_data = self.raw_data
             if len(self.raw_data) > self.data_len_max:
                 self.raw_data = self.raw_data[1:]

             if len(self.raw_data) > self.data_len_max//4:
                 # Define a bandpass filter, the parameters are the filter order, cutoff frequency, sampling frequency and type
                 b, a = butter(4, [0.8, 3], fs=self. frame_rate, btype="bandpass")
                 # Use the filtfilt function to filter the original data and return a filtered array
                 self.filtered_data = filtfilt(b, a, self.raw_data)

             # Use the Fourier transform to perform spectral analysis on the original data to obtain frequency and amplitude
             freqs = np.fft.rfftfreq(len(self.filtered_data), d=1/self.frame_rate)
             amps = np.abs(np.fft.rfft(self.filtered_data))
             amps = self. smooth_signal(amps)
             self.freq_x = freqs*60
             self.freq_y = amps

            # Find the frequency with the largest amplitude, as an estimate of heart rate, and convert to beats per minute
             heart_rate = np.max(freqs[np.argmax(amps)]) * 60
             # Add the heart rate value to the heart rate data array, and keep the length of the array not exceeding 100
             self.heart_rate_data = np.append(self.heart_rate_data, heart_rate)
             if len(self.heart_rate_data) > self.data_len_max//10:
                 self. heart_rate_data = self. heart_rate_data[1:]


     # Define a slot function that updates video frame and curve data, used to read the current frame of the camera and process and display it
     def update_frame(self):
         # Read a frame of image from the video capture object, return a boolean value and a numpy array
         ret, frame = self.cap.read()
         if not ret:
             return

             # If the read is successful, continue processing the image
         self.calc_heart_rate(frame)

        # In the slot function that updates the video frame and curve data, calculate the time and frame rate of the current frame, and update the time of the previous frame
         current_frame_time = time. time()
         frame_rate = 1 / (current_frame_time - self. last_frame_time)
         if len(self. frame_rate_list)>10:
             self.frame_rate_list = self.frame_rate_list[1:]
         self.frame_rate_list.append(frame_rate)
         self.frame_rate= sum(self.frame_rate_list)/len(self.frame_rate_list)
         self.last_frame_time = current_frame_time

         # Draw two numbers on the original image, representing the frame rate and heart rate respectively, using the putText function of opencv
         cv2.putText(frame, f"{self.frame_rate:.2f} fps", (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                     (255, 255, 255), 2)
         cv2.putText(frame, f"{self.heart_rate_data[-1]:.2f} bpm", (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                     (255, 255, 255), 2)

       # Convert the original image to an image in Qt format, and set the image of the label to this image
         qt_image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
         self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
         # Use pyqtgraph to draw raw data curves and heart rate curves, using different colors and styles
         self.raw_data_plot.clear()
         self. heart_rate_plot. clear()
         x=[ i/self. frame_rate for i in range(len(self. filtered_data))]
         self.raw_data_plot.plot(x, self.filtered_data, pen="g")#symbol="-",

         self.heart_rate_plot.plot(self.freq_x, self.freq_y, pen="r")
         self.heart_rate_plot.plot([self.heart_rate_data[-1],self.heart_rate_data[-1]],[0,50], pen="r")

         # Create an application object and pass in the command line parameters
if __name__ == "__main__":
     app = QApplication(sys.argv)
     # Create a window object and display it
     window = Ui_MainWindow()
     window. show()
     # Enter the main loop of the application and wait for user actions
     sys.exit(app. exec_())

