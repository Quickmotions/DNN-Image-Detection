import cv2
import argparse
import numpy as np

# cmd: python3 object_detection.py --image street.bmp --config yolov3.cfg --weights yolov3.weights --classes yolov3.txt
# pre-trained weights need to be installed from: https://pjreddie.com/media/files/yolov3.weights
# it would be best to have our own manually trainable model instead


class ObjectDetector:
    def __init__(self, image_path, config_path, weights_path, classes_path):
        self.image_path = image_path
        self.config_path = config_path
        self.weights_path = weights_path
        self.classes_path = classes_path
        self.classes = None
        self.load_classes()
        self.net = None
        self.scale = 0.00392
        self.conf_threshold = 0.5
        self.nms_threshold = 0.4
        self.COLORS = np.random.uniform(0, 255, size=(len(self.classes), 3))

    def load_classes(self):
        with open(self.classes_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]

    def load_model(self):
        self.net = cv2.dnn.readNet(self.weights_path, self.config_path)

    def detect_objects(self):
        image = cv2.imread(self.image_path)
        Width = image.shape[1]
        Height = image.shape[0]

        blob = cv2.dnn.blobFromImage(image, self.scale, (416, 416), (0, 0, 0), True, crop=False)
        self.net.setInput(blob)
        outs = self.net.forward(self.get_output_layers())

        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > self.conf_threshold:
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.conf_threshold, self.nms_threshold)

        for i in indices:
            try:
                box = boxes[i]
            except:
                i = i[0]

            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            self.draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x + w), round(y + h))

        cv2.imshow("object detection", image)
        cv2.waitKey()
        cv2.imwrite("output-file.jpg", image)
        cv2.destroyAllWindows()

    def get_output_layers(self):
        layer_names = self.net.getLayerNames()
        try:
            output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        except:
            output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]

        return output_layers

    def draw_prediction(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        label = str(self.classes[class_id])
        color = self.COLORS[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image', required=True, help='path to input image')
    parser.add_argument('-c', '--config', required=True, help='path to yolo config file')
    parser.add_argument('-w', '--weights', required=True, help='path to yolo pre-trained weights')
    parser.add_argument('-cl', '--classes', required=True, help='path to text file containing class names')
    args = parser.parse_args()

    detector = ObjectDetector(args.image, args.config, args.weights, args.classes)
    detector.load_model()
    detector.detect_objects()


if __name__ == "__main__":
    main()
