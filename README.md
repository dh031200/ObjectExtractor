## Object Extractor

### Installation
```bash
git clone https://github.com/dh031200/ObjectExtractor.git --recursive
cd ObjectExtractor
pip install -r requirements.txt
```

### Usage
```bash
python track.py --source sample.mp4 --model yolov8n.pt --save-vid
```

### Arguments
* `--source <video.mp4>` source video
* `--model <model.pt>` default: yolov8n.pt
* `--show` : display video while program is running
* `--save-vid` : save results video
* `--verbose` : print verbose