import argparse
import json
import logging
import math
import sys
import time
from collections import namedtuple

import coloredlogs
import cv2
import numpy as np
from klein import Klein

from tf_pose import common
from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path
from tf_pose.networks import model_wh

# define a handy tuple for coordinates
Point = namedtuple("Point", ["x", "y"])

logger = logging.getLogger("PoseEstimatorServer")
logger.handlers.clear()


def get_args():
    parser = argparse.ArgumentParser(description="tf-pose-estimation server")
    parser.add_argument(
        "--model",
        type=str,
        default="mobilenet_thin",
        help="cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small",
    )
    parser.add_argument(
        "--resize",
        type=str,
        default="0x0",
        help="if provided, resize images before they are processed. "
        "default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ",
    )
    parser.add_argument(
        "--resize-out-ratio",
        type=float,
        default=4.0,
        help="if provided, resize heatmaps before they are post-processed. default=1.0",
    )
    parser.add_argument("--port", "-p", type=int, default=5000, help="Server port")
    parser.add_argument(
        "--debug", "-d", action="store_true", help="Set logging to DEBUG"
    )

    args = parser.parse_args()
    return args


def plot_image(image):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    a = fig.add_subplot(1, 1, 1)
    a.set_title("Image")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()


class PoseEstimation:
    app = Klein()

    def __init__(self, model_name, resize_shape):
        try:
            self.model_name = model_name
            self.load_model(model_name, resize_shape)
        except Exception as e:
            logger.error("Error loading Pose Estimation model: {e}")
            logger.exception(e)

    def load_model(self, model_name, resize_shape):
        w, h = model_wh(resize_shape)
        if w == 0 or h == 0:
            w, h = (432, 368)

        self.w, self.h = w, h
        self.model = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))

    @staticmethod
    def decode_image(image_data):
        # convert string of image data to uint8
        nparr = np.fromstring(image_data, np.uint8)
        # decode image
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    @staticmethod
    def calculate_angles(humans, w, h):
        def pixel_pos(body_part):
            center = Point(int(body_part.x * w + 0.5), int(body_part.y * h + 0.5),)
            return center

        def angle(p1, p2):
            delata_x = p2.x - p1.x
            delata_y = p2.y - p1.y
            return math.degrees(math.atan(delata_y / delata_x))

        def middle_point(p1, p2):
            return Point((p2.x + p1.x) / 2, (p2.y + p1.y) / 2)

        inclinations = []
        confidences = []
        for i, human in enumerate(humans):

            try:
                logger.info(f"Processing detected human {i}")

                nose = pixel_pos(human.body_parts[0])
                l_hip = pixel_pos(human.body_parts[8])
                r_hip = pixel_pos(human.body_parts[11])

                logger.debug(f"Nose position: {nose}")
                logger.debug(f"Left Hip position: {l_hip}")
                logger.debug(f"Right Hip  position: {r_hip}")

                # calculate the hip center point
                hip_center = middle_point(l_hip, r_hip)
                logger.debug(f"Hip center position: {hip_center}")

                # calculate angle between nose and hip center w.r.t horizon
                inclination = angle(nose, hip_center)
                logger.debug(f"Person {i} inclination: {inclination}")

                inclinations.append(inclination)
                confidences.append(1.0)
            except Exception as e:
                inclinations.append(-1)
                confidences.append(0.0)
                logger.error(f"Error caculating inclination for person {i}: {e}")

        # TODO: Compute aggregate conficende
        return inclinations, confidences

    @app.route("/pose-estimation/angle", methods=["POST"])
    def inference(self, request):
        method = request.method.decode("utf-8").upper()
        content_type = request.getHeader("content-type")

        logger.debug(f"Method: {method}")
        logger.debug(f"Content_type: {content_type}")

        try:
            # read the image as an IO stream
            image = PoseEstimation.decode_image(request.content.read())
            w, h, c = image.shape
            logger.debug(f"Received image of size: {w}x{h}x{c}")

            # start inference on the image
            t = time.time()
            humans = self.model.inference(
                image,
                resize_to_default=(self.w > 0 and self.h > 0),
                upsample_size=args.resize_out_ratio,
            )
            elapsed = time.time() - t

            logger.info(
                f"Run inference with {self.model_name} in {elapsed:.4f} seconds."
            )

            angles, confidences = PoseEstimation.calculate_angles(humans, w, h)
            return json.dumps(
                [
                    {"person": i, "angle": angle, "confidence": confidences[i]}
                    for i, angle in enumerate(angles)
                ]
            )

        except Exception as e:
            logger.error(f"Error running pose estimation: {e}")
            logger.exception(e)
            return f"Error {e}"


if __name__ == "__main__":

    args = get_args()

    # Install console logger
    coloredlogs.install(
        fmt="%(levelname)s - %(filename)s - %(lineno)s: %(message)s",
        level=logging.DEBUG if args.debug else logging.INFO,
    )

    try:
        # create server and run
        pest = PoseEstimation(args.model, args.resize)
        pest.app.run("0.0.0.0", args.port)
    except Exception as e:
        logger.error(f"Unknown exception while starting Pose Estimation server: {e}")
        logger.exception(e)
