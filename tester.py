import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
from utils.helpers import to_cuda
from batchgenerators.utilities.file_and_folder_operations import *
import pandas as pd
class Tester(Trainer):
    def __init__(self,config, test_loader, model,is_2d, loss,  model_name):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.config = config
        self.test_loader = test_loader
        self.model = model
        self.is_2d = is_2d
        self.model_name = model_name
        self.save_path = "save_results/" + model_name
        dir_exists(self.save_path)
        remove_files(self.save_path)
        cudnn.benchmark = True

    def test(self):
        self.model.eval()
        self._reset_metrics()
        tbar = tqdm(self.test_loader, ncols=150)
        tic = time.time()
        with torch.no_grad():
            
            for i, (img, gt) in enumerate(tbar):
                self.data_time.update(time.time() - tic)
                img = to_cuda(img)
                gt = to_cuda(gt)
                if not self.is_2d:
                    img = img.unsqueeze(1)
                with torch.cuda.amp.autocast(enabled=self.config.AMP):
                    pre = self.model(img)
                    loss = self.loss(pre, gt)
                self.total_loss.update(loss.item())
                self.batch_time.update(time.time() - tic)
               
                gt = gt.squeeze(1)
                pre = pre.squeeze(1)
                
                predict = torch.sigmoid(pre).cpu().detach().numpy()
                predict_b = np.where(predict >= 0.5, 1, 0)
                for j in range(gt.shape[0]):
                    cv2.imwrite(self.save_path + f"/gt{i*gt.shape[0]+j}.png", np.uint8(gt[j].cpu().numpy()*255))
                    cv2.imwrite(self.save_path + f"/pre{i*gt.shape[0]+j}.png", np.uint8(predict[j]*255))
                    cv2.imwrite(self.save_path + f"/pre_b{i*gt.shape[0]+j}.png", np.uint8(predict_b[j]*255))
                    self._metrics_update(*get_metrics(pre[j], gt[j], 0.5).values())
                    self.CCC.update(count_connect_component(pre[j], gt[j], 0.5))
                tbar.set_description(
                    'TEST ({}) | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} Pre {:.4f} IOU {:.4f} |B {:.2f} D {:.2f} |'.format(
                        i, self.total_loss.average, *self._metrics_ave().values(), self.batch_time.average, self.data_time.average))
                tic = time.time()        
        data = list(self._metrics_ave().values())
        data.append(self.CCC.average)
        columns = list(self._metrics_ave().keys())
        columns.append("CCC")
        df = pd.DataFrame(data=np.array(data).reshape(1, len(columns)), index=[self.model_name], columns = columns)
        df.to_csv(join(self.save_path, f"{self.model_name}_result.cvs"))
        df.to_excel(join(self.save_path, f"{self.model_name}_result.xlsx"), sheet_name='CVSS')
        logger.info(f"###### TEST EVALUATION ######")
        logger.info(f'test time:  {self.batch_time.average}')
        logger.info(f'     loss:  {self.total_loss.average}')
        
        logger.info(f'     CCC:  {self.CCC.average}')
        for k, v in self._metrics_ave().items():
            logger.info(f'{str(k):5s}: {v}')

   
        