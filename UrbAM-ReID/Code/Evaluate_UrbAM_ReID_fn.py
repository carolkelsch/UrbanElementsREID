import os
import numpy as np
import xml.dom.minidom as XD
import xml.etree.ElementTree as ET
import os.path as osp
import argparse
import sys
from io import StringIO


def evaluate_reid(track_file, data_path):
    """
    Evaluates ReID performance and returns mAP and CMC values.

    Args:
        track_file (str): Path to the track list text file.
        data_path (str): Base path to the dataset folders and XML files.

    Returns:
        tuple: (mAP, CMC) where mAP is a float and CMC is a numpy array of rank-based match rates.
    """
    #function to read the xml file
    def read_xml(xml_dir, dir_path):
        xmlp = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(xml_dir,parser=xmlp)
        root = tree.getroot()
        xmlp = ET.XMLParser(encoding="utf-8")
        tree = ET.parse(xml_dir, parser=xmlp)
        root = tree.getroot()
        dataset=[]
        for element in root.iter('Item'):
            pid = int(element.get('objectID'))
            image_name = int(element.get('imageName')[:-4])
            dataset.append([pid,image_name])
        return dataset

    # Load ground truth data
    xml_dir = os.path.join(data_path, 'test_label.xml')
    dataset_gallery = read_xml(xml_dir, data_path)
    gallery_gt = np.array(dataset_gallery)

    xml_dir = os.path.join(data_path, 'query_label.xml')
    query_path = os.path.join(data_path, 'query_test')
    dataset_query = read_xml(xml_dir, query_path)
    query_gt = np.array(dataset_query)
    
    #read the track2 file
    with open(track_file) as f:
        line = f.readlines()
        line = [x.strip() for x in line]
        track2 = []
        for i in range(len(line)):
            track2.append(line[i].split(' '))
        track2 = np.array(track2)
        track2 = track2.astype(np.int32)


    #save the vehicle ID ground truth of the gallery
    id_gallery = gallery_gt[:,0]
    #save the vehicle ID ground truth of the gallery
    id_query = query_gt[:,0]

    AP=0.0
    CMC=np.zeros( len(track2))

    #Calculate AP and CMC for each query
    for j in range(len(query_gt)):

        query_id = query_gt[j,0]
        true_positives = np.where(id_gallery == query_id)
        if len(true_positives[0]) == 0:
            a=1
        index = track2[j,:] - 1
        sortID = id_gallery[index]

        sortIDGallery = sortID
        #find the good index
        rows_good = np.where(sortIDGallery == query_id)

        #initialize
        ap=0
        cmc=np.zeros( len(track2))

        #find the number of good index
        ngood = (true_positives[0].shape)[0]


        if rows_good[0].size != 0:

            cmc[rows_good[0][0]:] = 1

            for i in range(len(rows_good[0])):
                #recall
                d_recall = 1.0 / ngood
                #precision
                precision = (i+1) * 1.0 / (rows_good[0][i]+1)
                if rows_good[0][i] != 0:
                    old_precision =(i+1) * 1.0 / (rows_good[0][i]+1)
                else:
                    old_precision = 1.0
                ap = ap + d_recall * (old_precision + precision) / 2

        CMC = CMC + cmc
        AP = AP + ap

    #calculate de mean average precision
    mAP = AP / len(track2)
    #calculate the mean cumulative match characteristic
    CMC = CMC / len(track2)

    return mAP, CMC
