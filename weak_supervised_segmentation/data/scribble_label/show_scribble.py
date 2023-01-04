import cv2
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
def scribble_show(data_path, scribble_path,label_path, centerline_path,process_data_path, num_sequence = 30,is_overwrite = True) :
    
    if is_overwrite and isdir(process_data_path):
        shutil.rmtree(process_data_path)
    os.makedirs(process_data_path, exist_ok=True)  
    for id in range(num_sequence):
        img = cv2.imread(os.path.join(data_path, f"{id}.png"), 0)
        scr_lab = cv2.imread(os.path.join(scribble_path, f"{id}.PNG"), 0)
        lab = cv2.imread(os.path.join(label_path, f"label_s{id}.png"), 0)
        cen_lab = cv2.imread(os.path.join(centerline_path, f"label_s{id}.png"), 0)
        img = np.expand_dims(img, 2)
        img = np.repeat(img, 3, 2)
        img2 = img.copy()
        img3 = img.copy()
        # lab = np.where(lab >= 135, 255, 0)
        h,w = scr_lab.shape
        for i in range(h):
            for j in range(w):
                if scr_lab[i,j] >=80:
                    scr_lab[i,j] = 255
                    img[i,j,:]=[0,0,128]+img[i,j,:]*0.5
                    # img[i,j,:]=[0,107,128]+img[i,j,:]*0.5
                elif scr_lab[i,j] > 10:
                    scr_lab[i,j] = 100
                    img[i,j,:]=[0,128,128]+img[i,j,:]*0.5
                    
                else:
                    scr_lab[i,j] = 0
                if lab[i,j] >=110:
                    img2[i,j,:]=[0,0,128]+img2[i,j,:]*0.5
                    # img2[i,j,:]=[0,107,128]+img2[i,j,:]*0.5
    
                if cen_lab[i,j] == 1:
                    img3[i,j,:]=[0,0,255]
                    # img3[i,j,:]=[0,107,128]+img3[i,j,:]*0.5
                elif scr_lab[i,j] == 100:
                    img3[i,j,:]=[0,128,128]+img3[i,j,:]*0.5
                   
        print(f"{id} Done")

                

                
        cv2.imwrite(f"{process_data_path}/label_{id}.png", scr_lab)
        cv2.imwrite(f"{process_data_path}/scr_image_{id}.png", img)
        cv2.imwrite(f"{process_data_path}/image_{id}.png", img2)
        cv2.imwrite(f"{process_data_path}/cen_{id}.png", img3)
           
    

     
        
    # def save_seq_png(self,seqs_list, path):
    #     for id_s, seq in enumerate(seqs_list):
    #         for id_i,img in enumerate(seq):
    #             file = f"image_s{id_s}_i{id_i}.png"
    #             cv2.imwrite(f"{path}/{file}", img*255)
    #             print(f'save_seqs : {file}')

    # def save_lab_png(self,labs_list, path):
    #     for id_s, lab in enumerate(labs_list):
    #         file = f"label_s{id_s}.png"
    #         cv2.imwrite(f"{path}/{file}", lab)
    #         print(f'save_labs : {file}')

def main():
    data_path = "/home/lwt/data/CVSS/label/training/min"
    scribble_path = "/home/lwt/data/CVSS/label/training/hand-painted"
    label_path = "/home/lwt/data/CVSS/label/training/labels"
    process_data_path="/home/lwt/data/CVSS/label/training/scribble_show"
    centerline_path = "/home/lwt/data/CVSS/scribble/centerline_vessels"
    scribble_show(data_path,scribble_path,label_path,centerline_path,process_data_path)
  
    


        


if __name__ == '__main__':
    main()


def get_color(img, gt_bak, gt_scr):
    H, W,_ = img.shape
  
    for i in range(H):
        for j in range(W):
            # if gt_bak[i, j] == 255:
            #     img[i,j,:]=[0,0,128]+img[i,j,:]*0.5
            if gt_scr[i, j] == 255:
                img[i,j,:]=[128,0,0]+img[i,j,:]*0.5
           
    return img

# img = cv2.imread("scribble/img.png")
# gt_bak = cv2.imread("scribble/bak.PNG",0)
# gt_bak = np.where(gt_bak > 0, 255, 0).astype(np.uint8)
# gt_scr = cv2.imread("scribble/scr.PNG",0)
# gt_scr = np.where(gt_scr > 0, 255, 0).astype(np.uint8)
# cv2.imwrite("scribble/color.png", get_color(img, gt_bak, gt_scr))
# print(gt_bak.shape)




                