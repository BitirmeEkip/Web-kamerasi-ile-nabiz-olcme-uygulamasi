from kutuphane.kamera import Camera
from kutuphane.program import findFaceGetPulse
from kutuphane.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np
import datetime
#TODO: work on serial port comms, if anyone asks for it from serial import Serial
#Seri ithalattan herhangi biri isterse, seri port iletişimleri üzerinde çalışın Serial
import socket
import sys

class getPulseApp(object):

    """
    Python application that finds a face in a webcam stream, then isolates the
    forehead.

    Then the average green-light intensity in the forehead region is gathered
    over time, and the detected person's pulse is estimated.
    Bir web kamerası akışında bir yüz bulan ve ardından alnı izole eden Python uygulaması.
Daha sonra alın bölgesindeki ortalama yeşil ışık şiddeti zamanla toplanır ve tespit edilen kişinin nabzı tahmin edilir.
    """

    def __init__(self, args):
        # Imaging device - must be a connected camera (not an ip camera or mjpeg stream)
        # Görüntüleme cihazı - bağlı bir kamera olmalıdır (ip kamera veya mjpeg akışı değil)
        serial = args.serial
        baud = args.baud
        self.send_serial = False
        self.send_udp = False
        if serial:
            self.send_serial = True
            if not baud:
                baud = 9600
            else:
                baud = int(baud)
            self.serial = Serial(port=serial, baudrate=baud)

        udp = args.udp
        if udp:
            self.send_udp = True
            if ":" not in udp:
                ip = udp
                port = 5005
            else:
                ip, port = udp.split(":")
                port = int(port)
            self.udp = (ip, port)
            self.sock = socket.socket(socket.AF_INET, # Internet
                 socket.SOCK_DGRAM) # UDP

        self.cameras = []
        self.selected_cam = 0
        for i in range(3):
            camera = Camera(camera=i)  # first camera by default
            if camera.valid or not len(self.cameras):
                self.cameras.append(camera)
            else:
                break
        self.w, self.h = 0, 0
        self.pressed = 0
        # Containerized analysis of recieved image frames (an openMDAO assembly)
        # is defined next.

        # This assembly is designed to handle all image & signal analysis,
        # such as face detection, forehead isolation, time series collection,
        # heart-beat detection, etc.

        # Basically, everything that isn't communication
        # to the camera device or part of the GUI
        self.processor = findFaceGetPulse(bpm_limits=[50, 120],
                                          data_spike_limit=3000.,
                                          face_detector_smoothness=10.)

        # Init parameters for the cardiac data plot
        self.bpm_plot = False
        self.plot_title = "Data display - raw signal (top) ve PSD (bottom)"

        # Maps keystrokes to specified methods
        #(A GUI window must have focus for these to work)
        self.key_controls = {"s": self.toggle_search,
                             "d": self.toggle_display_plot,
                             "c": self.toggle_cam,
                             "f": self.write_csv}

    def toggle_cam(self):
        if len(self.cameras) > 1:
            self.processor.find_faces = True
            self.bpm_plot = False
            destroyWindow(self.plot_title)
            self.selected_cam += 1
            self.selected_cam = self.selected_cam % len(self.cameras)

    def write_csv(self):
        """
        Geçerli verileri bir csv dosyasına yazar
        """
        fn = "Webcam-nabiz" + str(datetime.datetime.now())
        fn = fn.replace(":", "_").replace(".", "_")
        data = np.vstack((self.processor.times, self.processor.samples)).T
        np.savetxt(fn + ".csv", data, delimiter=',')
        print("csv yazma")

    def toggle_search(self):
        """
        İşlemcinin yüz algılama bileşeninde bir hareket kilidini açar veya kapatır.
Alın konumunun yerinde kilitlenmesi, bir alın başarılı bir şekilde izole edildikten sonra veri kalitesini önemli ölçüde artırır.
        """
        #state = self.processor.find_faces.toggle()
        state = self.processor.find_faces_toggle()
        print("yuz algilama kilidi =", not state)

    def toggle_display_plot(self):
        """
        Toggles the data display.
        Veri gösterimini değiştirir.
        """
        if self.bpm_plot:
            print("bpm plot devre disi")
            self.bpm_plot = False
            destroyWindow(self.plot_title)
        else:
            print("bpm plot etkinlestirildi")
            if self.processor.find_faces:
                self.toggle_search()
            self.bpm_plot = True
            self.make_bpm_plot()
            moveWindow(self.plot_title, self.w, 0)

    def make_bpm_plot(self):
        """
        Creates and/or updates the data display
        Veri görüntüsünü oluşturur ve/veya günceller
        """
        plotXY([[self.processor.times,
                 self.processor.samples],
                [self.processor.freqs,
                 self.processor.fft]],
               labels=[False, True],
               showmax=[False, "bpm"],
               label_ndigits=[0, 0],
               showmax_digits=[0, 1],
               skip=[3, 3],
               name=self.plot_title,
               bg=self.processor.slices[0])

    def key_handler(self):
        """
        Handle keystrokes, as set at the bottom of __init__()

        A plotting or camera frame window must have focus for keypresses to be
        detected.
        __init__() öğesinin altında ayarlandığı gibi tuş vuruşlarını işleme
Tuşa basmaların algılanması için bir çizim veya kamera çerçevesi penceresinin odak olması gerekir.
        """

        self.pressed = waitKey(10) & 255  # 10 saniye tuşa basılmasını bekleyin
        if self.pressed == 27:  # 'esc' ile programdan çık
            print("Exiting")
            for cam in self.cameras:
                cam.cam.release()
            if self.send_serial:
                self.serial.close()
            sys.exit()

        for key in self.key_controls.keys():
            if chr(self.pressed) == key:
                self.key_controls[key]()

    def main_loop(self):
        """
        Single iteration of the application's main loop.
        Uygulamanın ana döngüsünün tek yinelemesi.
        """
        # Get current image frame from the camera
        # Kameradan geçerli görüntü çerçevesini alın.
        frame = self.cameras[self.selected_cam].get_frame()
        self.h, self.w, _c = frame.shape

        # display unaltered frame / değiştirilmemiş çerçeveyi göster
        # imshow("Original",frame)

        # set current image frame to the processor's input / mevcut görüntü çerçevesini işlemcinin girişine ayarla
        self.processor.frame_in = frame
        # process the image frame to perform all needed analysis /gerekli tüm analizleri gerçekleştirmek için görüntü çerçevesini işleyin
        self.processor.run(self.selected_cam)
        # collect the output frame for display / görüntü için çıktı çerçevesini topla
        output_frame = self.processor.frame_out

        # show the processed/annotated output frame / işlenmiş/açıklamalı çıktı çerçevesini göster
        imshow("Processed", output_frame)

        # create and/or update the raw data display if needed / gerekirse ham veri görüntüsünü oluşturun ve/veya güncelleyin
        if self.bpm_plot:
            self.make_bpm_plot()

        if self.send_serial:
            self.serial.write(str(self.processor.bpm) + "\r\n")

        if self.send_udp:
            self.sock.sendto(str(self.processor.bpm), self.udp)

        # handle any key presses / herhangi bir tuşa basmak
        self.key_handler()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Webcam nabız dedektoru.')
    parser.add_argument('--serial', default=None,
                        help='bpm verileri için seri bağlantı noktası hedefi')
    parser.add_argument('--baud', default=None,
                        help='Seri iletim için baud hızı')
    parser.add_argument('--udp', default=None,
                        help='udp adresi:bpm verileri için bağlantı noktası hedefi')

    args = parser.parse_args()
    App = getPulseApp(args)
    while True:
        App.main_loop()
