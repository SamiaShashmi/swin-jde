import os
import os.path as osp
import argparse

import numpy as np 

from kmeans import kmeans, avg_iou

NUM_ANCHORS = 12 

def read_bbox(split='VisDrone2019-MOT-train', if_certain_seq=False, if_norm=False):

    seqs_str = '''MOT17-02-SDP
                    MOT17-04-SDP
                    MOT17-05-SDP
                    MOT17-09-SDP
                    MOT17-10-SDP
                    MOT17-11-SDP
                    MOT17-13-SDP'''
    data_root = 'MOT17/labels_with_ids/train'

    seqs = [seq.strip() for seq in seqs_str.split()]

    if if_certain_seq:
        seq_list = certain_seqs
    else:
        seq_list = seqs

    bbox_wh = []

    for seq in seq_list:
        anno_files = os.listdir(osp.join(data_root, seq, 'img1'))
        for anno_file in anno_files:
            with open(osp.join(osp.join(data_root, seq, 'img1'), anno_file), 'r') as f:
                
                lines = f.readlines()

                for row in lines:
                    current_line = row.split(' ')

        
                    if seq in 'MOT17-05' or seq in 'MOT17-06':
                        orig_w, orig_h = 640, 480
                    else:
                        orig_w, orig_h = 1920, 1080
                    
                    bbox_wh.append([int(float(current_line[4]) * orig_w), int(float(current_line[5]) * orig_h)])

            f.close()

    
    return np.array(bbox_wh)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='VisDrone2019-MOT-train', help='train or test')
    parser.add_argument('--if_certain_seqs', type=bool, default=False, help='for debug')
    parser.add_argument('--if_norm', type=bool, default=False, help='if normalization')
    opt = parser.parse_args()
    
    bbox_wh = read_bbox(opt.split, opt.if_certain_seqs)

    print(bbox_wh.shape)

    out = kmeans(bbox_wh, NUM_ANCHORS)
    print("Accuracy: {:.2f}%".format(avg_iou(bbox_wh, out) * 100))
    print("Boxes:\n {}".format(out))

    ratios = np.around(out[:, 0] / out[:, 1], decimals=2).tolist()
    print("Ratios:\n {}".format(sorted(ratios)))