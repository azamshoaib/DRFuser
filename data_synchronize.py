#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import glob
import argparse
import time
import cv2
import pandas as pd
import sys
import subprocess
foo = subprocess.Popen("/home/shoaib/work/dvs_data/test.sh", shell=True, executable="/bin/bash")
import rosbag
from sensor_msgs.msg import Image
from autoware_can_msgs.msg import CanInfo
from cv_bridge import CvBridge
import bagpy
from bagpy import bagreader

def can_extract(bag_file_name):
    can_topic = "/can_info"
    b = bagreader(bag_file_name)
    print(b.topic_table)
    data = b.message_by_topic(can_topic)
    bag_dir = bag_file_name.split('.')
    data = pd.read_csv(os.path.join(bag_dir[0],'can_info.csv'))
    can_index = [x for x in range(0, len(data.values))]
    angle = data['angle']
    can_time = data['Time']
    candata = {'can_ids':can_index,'time_can':can_time,'angle':angle}
    can_data_frame = pd.DataFrame(candata)
    can_data_frame.to_csv(os.path.join(bag_dir[0],'can_info.csv'))
    return can_data_frame

def aps_extract(bag_file_name):
    bag_dir = bag_file_name.split('.')
    output_dir_aps = os.path.join(bag_dir[0],'aps_data/')
    os.mkdir(output_dir_aps)
    aps_topic="/camera1/usb_cam1/image_raw"
    count_aps = 0
    time_aps = []
    bag = rosbag.Bag(bag_file_name, "r")
    bridge = CvBridge()
    for topic1, msg1, t1 in bag.read_messages(topics=[aps_topic]):
        cv_aps = bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        time1 = float(t1.secs)
        if count_aps<10:
            cv2.imwrite(output_dir_aps+str('aps')+'_'+str('000000')+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        elif count_aps<100:
            cv2.imwrite(output_dir_aps+str('aps')+'_'+str('00000')+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        elif count_aps<1000:
            cv2.imwrite(output_dir_aps+str('aps')+'_'+str('0000')+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        elif count_aps<10000:
            cv2.imwrite(output_dir_aps+str('aps')+'_'+str('000')+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        elif count_aps<100000:
            cv2.imwrite(output_dir_aps+str('aps')+'_'+str('00')+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        elif count_aps<1000000:
            cv2.imwrite(output_dir_aps+str('aps')+'_'+str('0')+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        # cv2.imwrite(output_dir_aps+'_'+str(count_aps)+'_'+str(time1)+'.png', cv_aps)
        # print ("Wrote image %i" , count_aps)
        time_aps.append(time1)
        count_aps += 1
    
    aps_files = sorted(glob.glob(os.path.join(output_dir_aps,'*.png')))
    aps_index = [x for x in range(0, len(aps_files))]
    aps_dataframe = pd.DataFrame({"aps_index":aps_index,"time_aps":time_aps,"aps_file":aps_files})
    aps_dataframe.to_csv(os.path.join(bag_dir[0],'aps_data_day2-2-gist.csv'))
    bag.close()
    print('TOTAL_{0}_APS_FILES_DONE'.format(len(aps_files)))
    return aps_dataframe

def dvs_extract(bag_file_name):
    bag_dir = bag_file_name.split('.')
    output_dir_dvs = os.path.join(bag_dir[0],'dvs_data/')
    os.mkdir(output_dir_dvs)
    dvs_topic="/dvs_rendering"
    count_dvs = 0
    time_dvs = []
    bag = rosbag.Bag(bag_file_name, "r")
    bridge = CvBridge()
    for topic1, msg1, t1 in bag.read_messages(topics=[dvs_topic]):
        cv_dvs = bridge.imgmsg_to_cv2(msg1, desired_encoding="bgr8")
        time1 = float(t1.secs)
        if count_dvs<10:
            cv2.imwrite(output_dir_dvs+str('dvs')+'_'+str('000000')+str(count_dvs)+'_'+str(time1)+'.png', cv_dvs)
        elif count_dvs<100:
            cv2.imwrite(output_dir_dvs+str('dvs')+'_'+str('00000')+str(count_dvs)+'_'+str(time1)+'.png', cv_dvs)
        elif count_dvs<1000:
            cv2.imwrite(output_dir_dvs+str('dvs')+'_'+str('0000')+str(count_dvs)+'_'+str(time1)+'.png', cv_dvs)
        elif count_dvs<10000:
            cv2.imwrite(output_dir_dvs+str('dvs')+'_'+str('000')+str(count_dvs)+'_'+str(time1)+'.png', cv_dvs)
        elif count_dvs<100000:
            cv2.imwrite(output_dir_dvs+str('dvs')+'_'+str('00')+str(count_dvs)+'_'+str(time1)+'.png', cv_dvs)
        elif count_dvs<1000000:
            cv2.imwrite(output_dir_dvs+str('dvs')+'_'+str('0')+str(count_dvs)+'_'+str(time1)+'.png', cv_dvs)
        # print ("Wrote image %i" , count_dvs)
        time_dvs.append(time1)
        count_dvs += 1
    
    dvs_files = sorted(glob.glob(os.path.join(output_dir_dvs,'*.png')))
    dvs_index = [x for x in range(0, len(dvs_files))]
    dvs_dataframe = pd.DataFrame({"dvs_index":dvs_index,"time_dvs":time_dvs,"dvs_file":dvs_files})
    dvs_dataframe.to_csv(os.path.join(bag_dir[0],'dvs_data_day2-2-gist.csv'))
    bag.close()
    print('TOTAL_{0}_DVS_FILES_DONE'.format(len(dvs_files)))
    return dvs_dataframe

def sync_data_nn(bag_file_name,can_data,aps_data,dvs_data):
    bag_dir = bag_file_name.split('.')
    can_ids= can_data['can_ids'] 
    can_values = can_data['angle']
    can_time = can_data['time_can'].to_numpy()
    aps_time = aps_data['time_aps'].to_numpy()
    aps_index = aps_data['aps_index']
    aps_file = aps_data['aps_file']
    dvs_time = dvs_data['time_dvs'].to_numpy()
    dvs_index = dvs_data['dvs_index']
    dvs_file = dvs_data['dvs_file']

    dvs_out_sync= [min(range(len(dvs_time)), key=lambda dvs_idx: abs(dvs_time[dvs_idx]-ts)) for ts in aps_time]
    can_out_sync = [min(range(len(can_time)), key=lambda can_idx: abs(can_time[can_idx]-ts)) for ts in aps_time]
    dvs_out= pd.DataFrame(dvs_out_sync)
    dvs_out.to_csv(os.path.join(bag_dir[0],'dvs_out_sync.csv'))

    can_out= pd.DataFrame(can_out_sync)
    can_out.to_csv(os.path.join(bag_dir[0],'can_out_sync.csv'))
    data1_can = {'aps_index':aps_index,'can_index':can_out_sync}
    data2_can = {'can_index':can_ids,'can_data':can_values}
    df1_can = pd.DataFrame(data1_can)
    df2_can = pd.DataFrame(data2_can)

    can_aps_sync = pd.merge(df1_can, 
                        df2_can, 
                        on ='can_index', 
                        how ='inner')
    data1_dvs = {'aps_index':aps_index,'dvs_index':dvs_out_sync}
    data2_dvs = {'dvs_index':dvs_index,'dvs_file':dvs_file}
    df1_dvs = pd.DataFrame(data1_dvs)
    df2_dvs = pd.DataFrame(data2_dvs)
    dvs_aps_sync = pd.merge(df1_dvs, 
                      df2_dvs, 
                      on ='dvs_index', 
                      how ='inner')
    complete_data_sync = {'aps_index':aps_index,'aps_file':aps_file,'dvs_file':dvs_aps_sync['dvs_file'],'can_data':can_aps_sync['can_data']}
    complete_data_sync_dataframe = pd.DataFrame(complete_data_sync)
    complete_data_sync_dataframe.to_csv(os.path.join(bag_dir[0],'complete_data.csv'))


    


if __name__ == '__main__':

    bag_file_name = '/home/shoaib/work/dvs_data/day3-1.bag'
    can_data = can_extract(bag_file_name)
    aps_data = aps_extract(bag_file_name)
    dvs_data = dvs_extract(bag_file_name)
    sync_data_nn(bag_file_name,can_data,aps_data,dvs_data)
