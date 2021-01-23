#!/usr/bin/env python3
#
# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Script to run generic MobileNet based classification model.
Modified for traffic sign classification.
"""


import argparse
import time

from picamera import PiCamera, Color
from aiy.vision import inference
from aiy.leds import Leds
from aiy.vision.models import utils
from gpiozero import Button
from aiy.pins import BUTTON_GPIO_PIN
from gpiozero import LED
from aiy.pins import PIN_A
from aiy.pins import PIN_B
from aiy.pins import PIN_C


# Initialize LED (in the button on the top of AIY Google Vision box)
leds = Leds()
leds.update(Leds.rgb_off())


# Initialize the GPIO pins A,B,C
pin_A = LED(PIN_A)
pin_B = LED(PIN_B)
pin_C = LED(PIN_C)


# Colors used for LED at the top of Google Vision AIY kit
RED = (0xFF, 0x00, 0x00)
GREEN = (0x00, 0xFF, 0x00)
BLUE = (0x00, 0x00, 0xFF)
PURPLE = (0xFF, 0x00, 0xFF)


# Set status of GPIO pin
def pinStatus(pin,status,gpio_logic):
    if gpio_logic=='INVERSE':
        if status=='HIGH':
            pin.off()
        if status=='LOW':
            pin.on()
    else:
        if status=='HIGH':
            pin.on()
        if status=='LOW':
            pin.off()
 

"""
index  label           function        pin_A pin_B pin_C
0      stop            stop            0     0     0
1      left            turn 90 degree  0     1     0 
2      right           turn -90 degree 0     0     1
3      slow            slow speed      0     1     1
4      background      move forward    1     0     0
"""
# Convert the most likely result to 3 binary signal and sent it out
def send_signal_to_pins(resuilt0,gpio_logic):
    if 'stop' in result0:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
        leds.update(Leds.rgb_on(RED))
    elif 'left' in result0:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'HIGH',gpio_logic)
        leds.update(Leds.rgb_on(BLUE))
    elif 'right' in result0:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'HIGH',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
        leds.update(Leds.rgb_on(PURPLE))
    elif 'slow' in result0:
        pinStatus(pin_A,'LOW',gpio_logic)
        pinStatus(pin_B,'HIGH',gpio_logic)
        pinStatus(pin_C,'HIGH',gpio_logic)
        leds.update(Leds.rgb_on(GREEN))
    else:
        pinStatus(pin_A,'HIGH',gpio_logic)
        pinStatus(pin_B,'LOW',gpio_logic)
        pinStatus(pin_C,'LOW',gpio_logic)
        leds.update(Leds.rgb_off())
    time.sleep(1)


# Get label names from retrained_labels.txt
def read_labels(label_path):
    with open(label_path) as label_file:
        return [label.strip() for label in label_file.readlines()]


def get_message(result, threshold, top_k):
    if result:
        return 'Detecting:\n %s' % '\n'.join(result)

    return 'Nothing detected when threshold=%.2f, top_k=%d' % (threshold, top_k)


def process(result, labels, tensor_name, threshold, top_k):
    """Processes inference result and returns labels sorted by confidence."""
    # MobileNet based classification model returns one result vector.
    assert len(result.tensors) == 1
    tensor = result.tensors[tensor_name]
    probs, shape = tensor.data, tensor.shape
    assert shape.depth == len(labels)
    pairs = [pair for pair in enumerate(probs) if pair[1] > threshold]
    pairs = sorted(pairs, key=lambda pair: pair[1], reverse=True)
    pairs = pairs[0:top_k]
    return [' %s (%.2f)' % (labels[index], prob) for index, prob in pairs]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True,
        help='Path to converted model file that can run on VisionKit.')
    parser.add_argument('--label_path', required=True,
        help='Path to label file that corresponds to the model.')
    parser.add_argument('--input_height', type=int, required=True, help='Input height.')
    parser.add_argument('--input_width', type=int, required=True, help='Input width.')
    parser.add_argument('--input_layer', required=True, help='Name of input layer.')
    parser.add_argument('--output_layer', required=True, help='Name of output layer.')
    parser.add_argument('--num_frames', type=int, default=None,
        help='Sets the number of frames to run for, otherwise runs forever.')
    parser.add_argument('--input_mean', type=float, default=128.0, help='Input mean.')
    parser.add_argument('--input_std', type=float, default=128.0, help='Input std.')
    parser.add_argument('--input_depth', type=int, default=3, help='Input depth.')
    parser.add_argument('--threshold', type=float, default=0.1,
        help='Threshold for classification score (from output tensor).')
    parser.add_argument('--top_k', type=int, default=3, help='Keep at most top_k labels.')
    parser.add_argument('--preview', action='store_true', default=False,
        help='Enables camera preview in addition to printing result to terminal.')
    parser.add_argument('--show_fps', action='store_true', default=False,
        help='Shows end to end FPS.')
    parser.add_argument('--gpio_logic', default='NORMAL',
        help='Indicates if NORMAL or INVERSE logic is used in GPIO pins.')
    args = parser.parse_args()

    model = inference.ModelDescriptor(
        name='mobilenet_based_classifier',
        input_shape=(1, args.input_height, args.input_width, args.input_depth),
        input_normalizer=(args.input_mean, args.input_std),
        compute_graph=utils.load_compute_graph(args.model_path))
    labels = read_labels(args.label_path)

    with PiCamera(sensor_mode=4, resolution=(1640, 1232), framerate=30) as camera:
        if args.preview:
            camera.start_preview()

        with inference.CameraInference(model) as camera_inference:
            for result in camera_inference.run(args.num_frames):
                processed_result = process(result, labels, args.output_layer,
                                           args.threshold, args.top_k)
                send_signal_to_pins(processed_result[0], args.gpio_logic)
                message = get_message(processed_result, args.threshold, args.top_k)
                if args.show_fps:
                    message += '\nWith %.1f FPS.' % camera_inference.rate
                print(message)

                if args.preview:
                    camera.annotate_foreground = Color('black')
                    camera.annotate_background = Color('white')
                    # PiCamera text annotation only supports ascii.
                    camera.annotate_text = '\n %s' % message.encode(
                        'ascii', 'backslashreplace').decode('ascii')

        if args.preview:
            camera.stop_preview()


if __name__ == '__main__':
    main()
