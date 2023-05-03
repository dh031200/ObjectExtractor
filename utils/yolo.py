from pathlib import Path
from typing import Union

from ultralytics import YOLO
from utils.tracker import track


class yolo(YOLO):
    def __init__(self, model: Union[str, Path] = 'yolov8n.pt', task=None) -> None:
        super().__init__(model, task)


class Model:
    def __init__(self, data, model, verbose):
        self.data = data
        self.model = model
        self.verbose = verbose
        self.detector = None
        self.src = None
        self.results = None
        self.writer = None
        self.load_model()

    def load_model(self):
        self.detector = yolo(self.model)

    def set_dir(self):
        self.data.save_dir = self.detector.predictor.save_dir
        self.data.img_save_dir = self.data.save_dir / 'imgs'
        self.data.img_save_dir.mkdir(parents=True, exist_ok=True)

    def track(self):
        detect = track(model=self.detector, source=self.data.src, verbose=self.verbose, tracker="botsort.yaml",
                       stream=True,
                       project='runs')
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
