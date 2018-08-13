# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 10:33:47 2017

@author: Will
"""

import numpy as np

class IDtracker:
    
    def __init__(self):
        self.ids = {}
        self.next_id = 1
        self.rms_sum = []
        self.rms_num = []
        self.min_place = []
        self.min_value = []
        
    def index(self, cor_mat):
        for key, value in self.ids.items():
                if np.array_equal(value, cor_mat):
                    return key
        else:
            self.ids[self.next_id] = cor_mat
            self.next_id = self.next_id + 1
            self.rms_sum.append(0)
            self.rms_num.append(0)
            return self.next_id - 1
        
    def get_ids(self):
#        print(self.ids)
        return self.ids
        
    def look_up(self, key):
        return self.ids[key]
    
    def track_rms(self, mid, rms):
        self.rms_sum[mid-1] = self.rms_sum[mid-1] + rms
        self.rms_num[mid-1] = self.rms_num[mid-1] + 1
        
    def get_rms_mean(self):
        return np.asarray(self.rms_sum, dtype = float) / np.asarray(self.rms_num, dtype = float)
    
    def track_min(self, mp, mv):
        self.min_place.append(mp)
        self.min_value.append(mv)
        
    def get_mins(self):
        return (np.asarray(self.min_place), np.asarray(self.min_value))
    
    def clean(self):
        self.ids.clear()
        self.next_id = 1
        self.rms_sum.clear()
        self.rms_num.clear()