## Object Extractor

### Installation

```bash
git clone https://github.com/dh031200/ObjectExtractor.git --recursive
cd ObjectExtractor
pip install -r requirements.txt
```

### Usage

```bash
python track.py --source sample.mp4 --model yolov8n.pt --appeared 100 --save-vid
```

### Arguments

* `--source <video.mp4>` source video
* `--model <model.pt>` default: yolov8n.pt
* `--appeared <threshold>` capture threshold
* `--show` : display video while program is running
* `--save-vid` : save results video
* `--verbose` : print verbose