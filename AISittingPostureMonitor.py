
from car import Receiver
from car import FakeCar
from car import LocalCar
import RPi.GPIO as GPIO
import smbus

# 也可以使用FakeCar，防止车子乱跑
import threading
import argparse
import numpy as np
import os
import cv2
import time
from tensorflow.keras import layers, models, callbacks
from sklearn.utils import class_weight
from pathlib import Path
import datetime

# Config settings
image_dimensions = (224, 224)
epochs = 10
model_name = 'new_model.h5'
keyboard_spacebar = 32
training_dir = 'train'


# beep setting
class beep:
    def __init__(self):
        self.address = 0x20
        self.bus = smbus.SMBus(1)

    def beep_on(self):
        self.bus.write_byte(self.address, 0x7F & self.bus.read_byte(self.address))

    def beep_off(self):
        self.bus.write_byte(self.address, 0x80 | self.bus.read_byte(self.address))


class CarController(threading.Thread):
    def __init__(self, port):
        threading.Thread.__init__(self)
        self.device = "/dev/" + port

    def run(self):
        my_car = LocalCar(device=self.device)
        while True:
            cmd = input()
            if (cmd == 'W'):
                my_car.write_command("A")
            elif (cmd == 'A'):
                my_car.write_command("H")
            elif (cmd == 'S'):
                my_car.write_command("E")
            elif (cmd == 'D'):
                my_car.write_command("B")
            elif (cmd == 'P'):
                my_car.write_command("Z")
            else:
                my_car.write_command(cmd)


#threading.Thread
class LiveViewer(threading.Thread):
    def __init__(self, soundson):
        threading.Thread.__init__(self)
        self.soundson = soundson
        self.buzzer_assistant = beep()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

    def run(self):
        
        mymodel = models.load_model(model_name)

        # Video capture stuff
        videocapture = cv2.VideoCapture(0)
        if not videocapture.isOpened():
            raise IOError('Cannot open webcam')

        last_poor_time = datetime.datetime.now()
        while True:
            _, frame = videocapture.read()
            cv2.imwrite('curImage.png', frame)
            im_color = cv2.imread('curImage.png')
            im = cv2.cvtColor(im_color, cv2.COLOR_BGR2GRAY)
            
            ##
            faces = self.face_cascade.detectMultiScale(im,1.1,6,cv2.CASCADE_SCALE_IMAGE)
            foundFaces = False
            if len(faces) > 0:
                foundFaces = True
            for (x, y, w, h) in faces:
                cv2.rectangle(im_color, (x, y), (x+w, y+h), (255, 0, 0), 2)
            ##
            im = cv2.resize(im, image_dimensions)
            im = im / 255  # Normalize the image
            im = im.reshape(1, image_dimensions[0], image_dimensions[1], 1)
            if foundFaces:
                predictions = mymodel.predict(im)
                class_pred = np.argmax(predictions)
                conf = predictions[0][class_pred]

                if (self.soundson and class_pred == 1):
                    # If Poor posture then open the buzzer
                    cur_poor_time = datetime.datetime.now()
                    interval = cur_poor_time - last_poor_time
                    if (int(interval.seconds) > 5):
                        self.buzzer_assistant.beep_on()
                        time.sleep(1)
                        self.buzzer_assistant.beep_off()
                    last_poor_time = cur_poor_time
            
                im_color = cv2.resize(im_color, (800, 480), interpolation=cv2.INTER_AREA)
                im_color = cv2.flip(im_color, flipCode=1)  # flip horizontally
                
                if (class_pred == 1):
                    # Poor
                    im_color = cv2.putText(im_color, 'Poor posture', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255),
                                           thickness=3)
                else:
                    # Good
                    im_color = cv2.putText(im_color, 'Good posture', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0),
                                           thickness=2)

                msg = 'Probability {:.2f}%'.format(float(conf * 100))
                im_color = cv2.putText(im_color, msg, (15, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 255), thickness=2)
            else:
                im_color = cv2.resize(im_color, (800, 480), interpolation=cv2.INTER_AREA)
                im_color = cv2.flip(im_color, flipCode=1)  # flip horizontally
                msg = 'No Target'
                im_color = cv2.putText(im_color, msg, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0),
                                           thickness=3)
            cv2.imshow('Posture Monitor', im_color)
            cv2.moveWindow('Posture Monitor', 20, 20);
            key = cv2.waitKey(20)
            if key == ord('q'):
                break

        videocapture.release()
        cv2.destroyAllWindows()


def doliveview(soundson):
    device_input = input("please input USB port:")
    liveviewer = LiveViewer(soundson)
	##
    liveviewer.run()
	##
    carcontroller = CarController(device_input)
    liveviewer.start()
    carcontroller.start()


def do_capture_action(action_n, action_label):
    img_count = 0
    output_folder = '{}/action_{:02}'.format(training_dir, action_n)
    print('Capturing samples for {} into folder {}'.format(action_n, output_folder))
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # Video capture stuff
    videocapture = cv2.VideoCapture(0)
    if not videocapture.isOpened():
        raise IOError('ERROR: Webcam open error')

    while True:
        _, frame = videocapture.read()
        filename = '{}/{:08}.png'.format(output_folder, img_count)
        cv2.imwrite(filename, frame)
        img_count += 1
        key = cv2.waitKey(1000)
        cv2.imshow('', frame)

        if key == keyboard_spacebar:
            break

    # Clean up
    videocapture.release()
    cv2.destroyAllWindows()


def do_training():
    train_images = []
    train_labels = []
    class_folders = os.listdir(training_dir)

    class_label_indexer = 0
    for c in class_folders:
        print('Training with class {}'.format(c))
        for f in os.listdir('{}/{}'.format(training_dir, c)):
            im = cv2.imread('{}/{}/{}'.format(training_dir, c, f), 0)
            im = cv2.resize(im, image_dimensions)
            train_images.append(im)
            train_labels.append(class_label_indexer)
        class_label_indexer = class_label_indexer + 1

    train_images = np.array(train_images)
    train_labels = np.array(train_labels)

    indices = np.arange(train_labels.shape[0])
    np.random.shuffle(indices)
    images = train_images[indices]
    labels = train_labels[indices]
    train_images = np.array(train_images)
    train_images = train_images / 255  # Normalize image
    n = len(train_images)
    train_images = train_images.reshape(n, image_dimensions[0], image_dimensions[1], 1)

    class_weights = class_weight.compute_sample_weight('balanced', train_labels)

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(image_dimensions[0], image_dimensions[1], 1)))
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))
    

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.25))


    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(len(class_folders), activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(train_images, train_labels, epochs=epochs, class_weight=class_weights)
    model.save(model_name)


def main():
    parser = argparse.ArgumentParser(prog='Posture Monitor', description='Posture Monitor on RasberryPi',
                                     epilog='Have a good time! :)')
    parser.add_argument('--capture-good', help='collect the good posture before training and living',
                        action='store_true')
    parser.add_argument('--capture-poor', help='collect the poor posture before training nad living',
                        action='store_true')
    parser.add_argument('--train', help='train model with collected images before living', action='store_true')
    parser.add_argument('--live', help='open the webcam and detect the posture all time', action='store_true')
    parser.add_argument('--buzzer', help='activate the buzzer, ONLY used with --live', action='store_true')
    args = parser.parse_args()

    if args.train:
        do_training()
    elif args.live:
        doliveview(args.buzzer)
    elif args.capture_good:
        do_capture_action(1, 'Good')
    elif args.capture_poor:
        do_capture_action(2, 'Poor')
    else:
        parser.print_help()


if __name__ == '__main__':
    main()




