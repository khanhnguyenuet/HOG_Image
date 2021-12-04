import argparse

from hog_feature import HOGFeature
from object_detector import ObjectDetector
from nms import NMS

def parse_args():
    parser = argparse.ArgumentParser("HOG demo")
    parser.add_argument('--inp', default='', type=str, help='Input Images')
    parser.add_argument('--model', '-m', default='', type=str, help='Input model')

    return parser.parse_args()

def main():
    args = parse_args()
    hog = HOGFeature()
    nms = NMS()
    image_path = args.inp
    model = args.model
    obj_detector = ObjectDetector(image_path, hog, model, nms)
    obj_detector()

if __name__ == "__main__":
    main()