import cv2
import os
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
import shutil
def scribble_show( scribble_path, luo_bak_path, centerline_path,process_data_path, num_sequence = 30, ignore_index = 255,is_overwrite = True) :

    
    cen_path = join(process_data_path,"centerline")
    scr_path = join(process_data_path,"scribble")
    if is_overwrite and isdir(cen_path):
        shutil.rmtree(cen_path)
    if is_overwrite and isdir(scr_path):
        shutil.rmtree(scr_path)
    os.makedirs(cen_path, exist_ok=True)  
    os.makedirs(scr_path, exist_ok=True)  
    for id in range(num_sequence):
       
        scr_lab = cv2.imread(os.path.join(scribble_path, f"{id}.PNG"), 0)
        cen_lab = cv2.imread(os.path.join(centerline_path, f"label_s{id}.png"), 0)
        luo_lab = cv2.imread(os.path.join(luo_bak_path, f"label_s{id}.png"), 0)
        h,w = scr_lab.shape
        new_scr = np.ones((h,w))*ignore_index
        new_cen = np.ones((h,w))*ignore_index
       
        h,w = scr_lab.shape
        for i in range(h):
            for j in range(w):
                if scr_lab[i,j] >=80:
                    new_scr[i,j] = 1
                   
                   
                # elif scr_lab[i,j] > 10:
                elif luo_lab[i,j] == 0:
                    new_scr[i,j] = 0
                    new_cen[i,j] = 0
                if cen_lab[i,j] ==1:
                    new_cen[i,j] = 1


        
      
               
                   
        print(f"{id} Done")

                

        cv2.imwrite(f"{cen_path}/label_s{id}.png", new_cen)
        cv2.imwrite(f"{scr_path}/label_s{id}.png", new_scr)     

   
           
    

     
        


def main():
   
    scribble_path = "/home/lwt/data/CVSS/training/hand-painted"
    luo_bak_path = "/home/lwt/data/CVSS/training/luo_labels"
    
   
    centerline_path = "/home/lwt/data/CVSS/scribble/centerline_vessels"
    process_data_path = "/home/lwt/data/CVSS/training/scribble_type"
    scribble_show(scribble_path,luo_bak_path,centerline_path,process_data_path)
  
    


        


if __name__ == '__main__':
    main()






                