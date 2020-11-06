from typing import List

import cv2 as cv2
import numpy as np

import gesture_commands
from realtimenet import camera
from realtimenet import engine
from realtimenet import feature_extractors
from realtimenet.downstream_tasks.gesture_recognition import INT2LAB
from realtimenet.downstream_tasks.nn_utils import LogisticRegression
from realtimenet.downstream_tasks.nn_utils import Pipe
from realtimenet.downstream_tasks.postprocess import PostprocessClassificationOutput


class GestureController:
    feature_extractor_weights = 'resources/backbone/strided_inflated_efficientnet.ckpt'
    gesture_classifier_weights = 'resources/gesture_detection/efficientnet_logistic_regression.ckpt'

    def __init__(self, commands: List[gesture_commands.BaseCommand]):
        # Initialize attributes
        self.net = None
        self.inference_engine = None
        self.frame_grabber = None
        self.postprocessors = None
        self.commands = commands

        self.setup_inference()

    def setup_inference(self):
        print("Setting up the inference")
        self._load_net()
        self._setup_inference_engine()
        self._setup_post_processor()

    def _load_net(self):
        # Load feature extractor
        feature_extractor = feature_extractors.StridedInflatedEfficientNet()
        checkpoint = engine.load_weights(self.feature_extractor_weights)
        feature_extractor.load_state_dict(checkpoint)
        feature_extractor.eval()

        # Load a logistic regression classifier
        gesture_classifier = LogisticRegression(num_in=feature_extractor.feature_dim, num_out=30)
        checkpoint = engine.load_weights(self.gesture_classifier_weights)
        gesture_classifier.load_state_dict(checkpoint)
        gesture_classifier.eval()

        # Combine both models
        self.net = Pipe(feature_extractor, gesture_classifier)

    def _setup_inference_engine(self):
        self.inference_engine = engine.InferenceEngine(self.net, use_gpu=True)
        video_source = camera.VideoSource(camera_id=0, size=self.inference_engine.expected_frame_size)
        self.frame_grabber = camera.VideoStream(video_source, self.inference_engine.fps)

    def _setup_post_processor(self):
        self.postprocessors = [
            PostprocessClassificationOutput(INT2LAB, smoothing=4)
        ]

    def run_inference(self):
        print("Starting inference")
        clip = np.random.randn(1, self.inference_engine.step_size, self.inference_engine.expected_frame_size[0],
                               self.inference_engine.expected_frame_size[1], 3)
        frame_index = 0
        display_error = None

        # Start threads
        self.inference_engine.start()
        self.frame_grabber.start()

        while True:
            try:
                frame_index += 1

                # Grab frame if possible
                img_tuple = self.frame_grabber.get_image()
                # If not possible, stop
                if img_tuple is None:
                    break

                # Unpack
                img, numpy_img = img_tuple

                clip = np.roll(clip, -1, 1)
                clip[:, -1, :, :, :] = numpy_img

                if frame_index == self.inference_engine.step_size:
                    # A new clip is ready
                    self.inference_engine.put_nowait(clip)

                frame_index = frame_index % self.inference_engine.step_size

                # Get predictions
                prediction = self.inference_engine.get_nowait()

                post_processed_data = {}
                for post_processor in self.postprocessors:
                    post_processed_data.update(post_processor(prediction))

                for command in self.commands:
                    command(post_processed_data)

            # Press escape to exit
            except KeyboardInterrupt:
                break
        # Clean up
        cv2.destroyAllWindows()
        self.frame_grabber.stop()
        self.inference_engine.stop()
