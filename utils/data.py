import time
import json
from pathlib import Path

import cv2

from utils.tracker import capture


class Data:
    def __init__(self, args):
        self.src = args.source
        self.show = args.show
        self.save_vid = args.save_vid

        self.tracker = None
        self.classes = None

        self.name = None
        self.width = None
        self.height = None
        self.total_frame = None
        self.fps = None
        self.writer = None
        self.save_dir = None
        self.img_save_dir = None

        self.capture = None
        self.way_info = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def get_src_info(self, dataset, result):
        self.name = Path(self.src).stem + '.mp4'
        self.width = result.orig_shape[1]
        self.height = result.orig_shape[0]

        if type(dataset.frames) == type(list()):
            self.total_frame = int(dataset.frames[0])
            self.fps = int(dataset.fps[0])
        else:
            self.total_frame = int(dataset.frames)
            self.fps = int(dataset.cap.get(cv2.CAP_PROP_FPS))

        assert self.total_frame, 'Invalid source'

    def set_writer(self):
        self.writer = cv2.VideoWriter(str(self.save_dir / self.name), cv2.VideoWriter_fourcc(*'mp4v'), self.fps,
                                      (self.width, self.height))

    def write(self, img):
        if img is not None:
            self.writer.write(img)

    def save(self, detect):
        if self.save_vid or self.show:
            output = detect.plot()
            if self.show:
                show_vid(output)
            if self.save_vid:
                self.write(output)
        save_track(self.tracker, detect.orig_img, self.img_save_dir, self.classes)

    def parse_detect_results(self):
        captured_list = []
        for track in self.tracker.captured_track:
            cls = self.classes[int(track.cls)]
            captured_list.append(
                dict(
                    id=track.track_id,
                    name=cls,
                    start_frame=track.start_frame,
                    end_frame=track.end_frame,
                    start_time=frame_to_time(track.start_frame, self.fps),
                    end_time=frame_to_time(track.end_frame, self.fps),
                    captured_time=frame_to_time(track.captured_frame, self.fps)
                )
            )
        self.capture = captured_list

    def save_results(self):
        with open(f'{self.save_dir}/result.json', 'w') as f:
            json.dump(dict(video_name=self.name, capture=self.capture), f, indent=4, ensure_ascii=False)

    def release(self):
        self.writer.release()


def save_img(path, img):
    cv2.imwrite(path, img)
    print('Image saved :', path)


def show_vid(img):
    cv2.imshow('Show', img)
    cv2.waitKey(1)


def save_track(tracker, img, path, names):
    capture_objects = capture(tracker)
    if capture_objects:
        for tid, cls, obj in capture_objects:
            l, t, r, b = map(int, obj)
            save_img(f'{path}/{str(tid).zfill(4)}_{names[int(cls)]}.png', img[t:b, l:r])


def frame_to_time(frame, fps=30):
    sec = frame // fps
    return time.strftime('%H:%M:%S', time.gmtime(sec))
