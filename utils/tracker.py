from ultralytics.yolo.utils import IterableSimpleNamespace, yaml_load
from ultralytics.tracker.trackers.bot_sort import BOTSORT, BOTrack
from ultralytics.tracker.track import on_predict_postprocess_end
from ultralytics.yolo.utils.checks import check_yaml


class BTrack(BOTrack):
    def __init__(self, tlwh, score, cls, feat=None, feat_history=50):
        super().__init__(tlwh, score, cls, feat, feat_history)
        self.captured_frame = None


class BTracker(BOTSORT):
    def __init__(self, args, frame_rate=30):
        super().__init__(args, frame_rate)
        self.captured_id = []
        self.captured_track = []

    def init_track(self, dets, scores, cls, img=None):
        if len(dets) == 0:
            return []
        if self.args.with_reid and self.encoder is not None:
            features_keep = self.encoder.inference(img, dets)
            return [BTrack(xyxy, s, c, f) for (xyxy, s, c, f) in zip(dets, scores, cls, features_keep)]  # detections
        else:
            return [BTrack(xyxy, s, c) for (xyxy, s, c) in zip(dets, scores, cls)]  # detections

    def multi_predict(self, tracks):
        BTrack.multi_predict(tracks)


TRACKER_MAP = {'botsort': BTracker}


def on_predict_start(predictor):
    tracker = check_yaml(predictor.args.tracker)
    cfg = IterableSimpleNamespace(**yaml_load(tracker))
    assert cfg.tracker_type in ['botsort'], f"Only support 'botsort' for now, but got '{cfg.tracker_type}'"
    trackers = []
    for _ in range(predictor.dataset.bs):
        tracker = TRACKER_MAP[cfg.tracker_type](args=cfg, frame_rate=30)
        trackers.append(tracker)
    predictor.trackers = trackers


def register_tracker(model):
    model.add_callback('on_predict_start', on_predict_start)
    model.add_callback('on_predict_postprocess_end', on_predict_postprocess_end)


def track(model, source=None, stream=False, **kwargs):
    if not hasattr(model.predictor, 'trackers'):
        register_tracker(model)
    conf = kwargs.get('conf') or 0.1
    kwargs['conf'] = conf
    kwargs['mode'] = 'track'
    return model.predict(source=source, stream=stream, **kwargs)


def capture(tracker, appeared=100):
    capture_objects = []
    for strack in tracker.tracked_stracks:
        if (tracker.frame_id - strack.start_frame > appeared) and (strack.track_id not in tracker.captured_id):
            capture_objects.append([strack.track_id, strack.cls, strack.tlbr])
            strack.captured_frame = tracker.frame_id
            tracker.captured_id.append(strack.track_id)
            tracker.captured_track.append(strack)
    return capture_objects
