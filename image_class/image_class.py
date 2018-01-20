

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from time import gmtime, strftime

import os
#import Image


"""
Author: Jeffrey Valentic
Date: 1/20/2018
File: image_class.py
Version 1.0
Notes:
    -This file should be in the same directory as 3 folders:
        1) error_log
        2) flagged_images
        3) image_evidence
    -This code was written to be implemented by the "Death Perception" crime alert system.
    -This script is for local use only, updates for network capabilities possible.
    
    
    
vCase Class (Violence Case) Details
dangerous(bool d): Handles cases where a situation is violent or non-violent.
    Non-Violent:
        photo will be moved to error_log for further training
    Violent:
        photo will be moved to image_evidence for evidence in a legal case
        police will be notified
        
send_alert():
    Prints date and time image was taken
    Asks user if it is criminal (y/n)
    returns true (y) or false (n)
    
"""

class vCase:
    
    evidence_path = ""
    error_path = ""
    file_path = ""
    image_name = ""
    final_path = ""
    _danger = True
    incident_time = None
    resolved = False
    
    def __init__(self, fp, img_name):
        self.file_path = fp
        self.image_name = img_name
        self.final_path = self.file_path + self.image_name
        self.error_path = "./error_log/" + self.image_name
        self.evidence_path = "./image_evidence/" + self.image_name
        self.incident_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    
        self._danger = self.send_alert()
        self.resolved = self.dangerous(self._danger)

    def dangerous(self, d):
        self._danger = d
        if self._danger == False:
            os.rename(self.final_path, self.error_path)
            print("image migrated to error_log folder")
            return True
        elif self._danger == True:
            os.rename(self.final_path, self.evidence_path)
            print("")
            print("Image has been stored in image_evidence folder.")
            print("Police have been notified.")
            print("")
            return True
    

    def send_alert(self):
        print("")
        print("Photo captured on: " + self.incident_time)
        image = mpimg.imread(self.final_path)
        plt.axis("off")
        plt.imshow(image)
        plt.show()
        danger = input("Confirm criminal activity (y/n): ")
        if danger == "y":
            return True
        elif danger == "n":
            return False

