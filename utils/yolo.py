from pathlib import Path
from typing import Union
from yaml import safe_load

from ultralytics import YOLO
from utils.tracker import track


class yolo(YOLO):
    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        super().__init__(model, task)


class Model:
    def __init__(self, data, model, verbose):
        self.conf = 0.25
        self.iou = 0.7
        self.half = False
        self.device = None
        self.save_dir = 'runs'
        self.show_conf = True
        self.show_labels = True
        self.show_boxes = True
        self.line_width = 0.0
        self.classes = None

        with open('config.yaml', 'r') as f:
            cfg = safe_load(f.read())


        allowed_keys = list(self.__dict__.keys())
        self.__dict__.update((key, value) for key, value in cfg.items()
                             if key in allowed_keys)
        rejected_keys = set(cfg.keys()) - set(allowed_keys)
        if rejected_keys:
            raise ValueError("Invalid arguments in constructor:{}".format(rejected_keys))

        self.data = data
        self.model = model
        self.verbose = verbose
        self.detector = None
        self.load_model()

    def load_model(self):
        self.detector = yolo(self.model)

    def set_dir(self):
        self.data.save_dir = self.detector.predictor.save_dir
        self.data.img_save_dir = self.data.save_dir / 'imgs'
        self.data.img_save_dir.mkdir(parents=True, exist_ok=True)

    def track(self):
        detect = track(model=self.detector, source=self.data.src, verbose=self.verbose, tracker="botsort.yaml",
                       stream=True, conf=self.conf, iou=self.iou, half=self.half, device=self.device,
                       project=self.save_dir, show_conf=self.show_conf, show_labels=self.show_labels,
                       boxes=self.show_boxes, classes=self.classes)
        self.model_init(self.detector, detect)
        self.set_dir()
        if self.data.save_vid:
            self.data.set_writer()
        return detect

    def model_init(self, model, result):
        res = next(result)
        self.data.get_src_info(model.predictor.dataset, res)
        setattr(self.data, 'tracker', model.predictor.trackers[0])
        setattr(self.data, 'classes', model.predictor.model.names)
