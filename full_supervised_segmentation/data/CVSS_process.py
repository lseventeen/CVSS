import os
import cv2
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import shutil

class CVSS_process(object):
    def __init__(self, data_path, process_data_path, new_slice = 8,num_sequence = 60,is_overwrite = True) -> None:
        self.data_path = data_path
        self.process_data_path = process_data_path
        self.num_sequence = num_sequence
        self.new_slice = new_slice

        if is_overwrite and isdir(self.process_data_path):
            shutil.rmtree(self.process_data_path)
        self.training_images_path = os.path.join(process_data_path, "training", "images")
        self.training_labels_path = os.path.join(process_data_path, "training", "labels")
        self.test_images_path = os.path.join(process_data_path, "test", "images")
        self.test_labels_path = os.path.join(process_data_path, "test", "labels")
        self.val_images_path = os.path.join(process_data_path, "validation", "images")
        self.val_labels_path = os.path.join(process_data_path, "validation", "labels")

        os.makedirs(self.training_images_path, exist_ok=True)
        os.makedirs(self.training_labels_path, exist_ok=True)
        os.makedirs(self.test_images_path, exist_ok=True)
        os.makedirs(self.test_labels_path, exist_ok=True)
        os.makedirs(self.val_images_path, exist_ok=True)
        os.makedirs(self.val_labels_path, exist_ok=True)

    def process(self):
        label_path = os.path.join(self.data_path, "labels")
        image_path = os.path.join(self.data_path, "images")
    
        image_files = list(sorted(os.listdir(image_path)))
        label_files = list(sorted(os.listdir(label_path)))
        slice_count = []
        sequences_list = []
        for i in range(1,self.num_sequence+1):
            slice_count_each_sequence = 0
            image_each_slice = []
            for j in image_files:
                if int(j[:2]) == i:
                    slice_count_each_sequence += 1
                    img = cv2.imread(os.path.join(image_path, j), 0)
                    image_each_slice.append(img)
                    print(j)
            slice_count.append(slice_count_each_sequence)
            sequences_list.append(np.array(image_each_slice))
        h,w = sequences_list[0].shape[1:]
        new_shape = [self.new_slice,h,w]
        image_list = []
        for s in sequences_list:
            sequence = resize(s,new_shape,order=3,mode = "edge",anti_aliasing=False)
            # mn = sequence.mean()
            # std = sequence.std()
            # print(sequence.shape, sequence.dtype, mn, std)
            # sequence = (sequence - mn) / (std + 1e-8)
            # image_full.append(ToTensor()(sequence))
            image_list.append(sequence)
        
        label_list = []
        for i in range(1,self.num_sequence+1):
            label_slice = []
            for j in label_files:
                if int(j[:2]) == i:
                    label = cv2.imread(os.path.join(label_path, j), 0)
                    label_slice.append(np.where(label >= 100, 255, 0).astype(np.uint8))
                    print(j)
            label = np.array(label_slice).max(axis=0)
            label_list.append(label)
        train_seq, test_seq, train_lab, test_lab = train_test_split(image_list, label_list, test_size = 1/3, random_state = 0)
        train_seq, val_seq, train_lab, val_lab = train_test_split(train_seq,train_lab, test_size=0.25, random_state=0)
        self.save_seq_png(train_seq,self.training_images_path)
        self.save_seq_png(test_seq,self.test_images_path)
        self.save_seq_png(val_seq,self.val_images_path)

        self.save_lab_png(train_lab,self.training_labels_path)
        self.save_lab_png(test_lab,self.test_labels_path)
        self.save_lab_png(val_lab,self.val_labels_path)
        
    def save_seq_png(self,seqs_list, path):
        for id_s, seq in enumerate(seqs_list):
            for id_i,img in enumerate(seq):
                file = f"image_s{id_s}_i{id_i}.png"
                cv2.imwrite(f"{path}/{file}", img*255)
                print(f'save_seqs : {file}')

    def save_lab_png(self,labs_list, path):
        for id_s, lab in enumerate(labs_list):
            file = f"label_s{id_s}.png"
            cv2.imwrite(f"{path}/{file}", lab)
            print(f'save_labs : {file}')


class CVSS_unlabel_process(CVSS_process):
    def __init__(self, data_path, process_data_path, new_slice = 8,num_sequence = 60,is_overwrite = True) -> None:
        self.data_path = data_path
        self.process_data_path = process_data_path
        self.num_sequence = num_sequence
        self.new_slice = new_slice

        if is_overwrite and isdir(self.process_data_path):
            shutil.rmtree(self.process_data_path)
        

       
        os.makedirs(self.process_data_path, exist_ok=True)

    def process(self):
        
    
        image_files = list(sorted(os.listdir(self.data_path)))
      
        slice_count = []
        sequences_list = []
        for i in range(1,self.num_sequence+1):
            slice_count_each_sequence = 0
            image_each_slice = []
            for j in image_files:
                if int(j[:2]) == i:
                    slice_count_each_sequence += 1
                    img = cv2.imread(os.path.join(self.data_path, j), 0)
                    image_each_slice.append(img)
                    print(j)
            slice_count.append(slice_count_each_sequence)
            sequences_list.append(np.array(image_each_slice))
        h,w = sequences_list[0].shape[1:]
        new_shape = [self.new_slice,h,w]
        image_list = []
        for s in sequences_list:
            sequence = resize(s,new_shape,order=3,mode = "edge",anti_aliasing=False)
            # mn = sequence.mean()
            # std = sequence.std()
            # print(sequence.shape, sequence.dtype, mn, std)
            # sequence = (sequence - mn) / (std + 1e-8)
            # image_full.append(ToTensor()(sequence))
            image_list.append(sequence)
        
        
        self.save_seq_png(image_list,self.process_data_path)
       
        
    

def main():
    data_path = "/home/lwt/data/CVSS/DSA"
    process_data_path="/home/lwt/data/CVSS/label"
    # data_path = "/home/lwt/data/CVSS/DSA_new"
    # process_data_path="/home/lwt/data/CVSS/unlabel"
    dp = CVSS_process(data_path, process_data_path)
    # dp = CVSS_unlabel_process(data_path, process_data_path)
    dp.process()
    


        


if __name__ == '__main__':
    main()

   