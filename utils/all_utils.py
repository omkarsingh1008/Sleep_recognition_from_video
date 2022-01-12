import requests
import os
import platform 
import subprocess
import cv2
from datetime import datetime
def get_media_path():
    """ it will return media path

    Returns:
        str: media path
    """
    r = requests.get("http://127.0.0.1:5000/get_media_path/")
    return r.json()['media_path']
def get_initialize_cams(number_of_camera = 7,use_camera=True):
    """ initialize all camera

    Args:
        number_of_camera (int, optional): [description]. Defaults to 7.

    Returns:
        list : in list cap object 
    """
    cap_list=[]
    if use_camera:
        for _ in range(1,number_of_camera+1):
            cap= cv2.VideoCapture()
            cap_list.append(cap)
    else:
        cap1=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000004226000000.mp4")
        cap2=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000004888000100.mp4")
        cap3=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000004987000000.mp4")
        cap4=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000005012000000.mp4")
        cap5=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000005066000100.mp4")
        cap6=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000005142000000.mp4")
        cap7=cv2.VideoCapture(r"D:\Lakshit\Duke\dukeplasto\full_video\00000005143000100.mp4")
        cap_list.append(cap1)
        cap_list.append(cap2)
        cap_list.append(cap3)
        cap_list.append(cap4)
        cap_list.append(cap5)
        cap_list.append(cap6)
        cap_list.append(cap7)
    return cap_list

def ping_ip(host=""):
    """it will ping ip

    Args:
        host (str, optional): [description]. Defaults to "".

    Returns:
        bool: True,False
    """

    param = '-n' if platform.system().lower()=='windows' else '-c'

    # Building the command. Ex: "ping -c 1 google.com"
    command = ['ping', param, '1', host]

    return subprocess.call(command) == 0

def check_server(address="http://127.0.0.1:5000/"):
    """it will check server is working of not

    Args:
        address (str, optional): [description]. Defaults to "http://127.0.0.1:5000/".

    Returns:
        int: status code
    """
    page = requests.get(address)
    return page.status_code 
def get_new_time():
    """for new time

    Returns:
        int: it will return tine in seconds
    """
    r = requests.get("http://127.0.0.1:5000/get_alarm_time/")
    SLEEP_TIME = r.json()["sleeptime"]
    SLEEP_TIME = (SLEEP_TIME*60)
    GUARD_NOT_FOUND_TIME = r.json()["misstime"]
    GUARD_NOT_FOUND_TIME = (GUARD_NOT_FOUND_TIME*60)

    return SLEEP_TIME, GUARD_NOT_FOUND_TIME
def getdatetime():
    dt = str(datetime.now())
    ts=dt.split(".")[0]
    ts=ts.replace(" ","_")
    ts=ts.replace(":","_")
    return ts









