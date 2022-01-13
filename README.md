# Sleep recognition from video
Sleep recognition based on motion detection.

## about the project

* project is divided into two main componets :- 
                                              1. Guard detection :- For guard detection, I use yolov5 model to detect a guard from live cctv feed and localize a guard. 
                                              2. sleep recognition :- For sleep recognition, I use motion detection if model detect any motion in guard bbox that mean guard is not sleeping but, if some priod of time no motion detect in bbox that's mean the guard is sleeping 

![alt text](https://github.com/omkarsingh1008/Sleep_recognition_from_video/blob/main/Screenshot%20from%202022-01-12%2015-43-02.png)


## Local installation
1. clone git repository
```bash
https://github.com/omkarsingh1008/Sleep_recognition_from_video.git
```
2. install the packages
```bash
pip install -r requirements.tx
```
4.at last, push in the command
```bash
python sleep.py
```

