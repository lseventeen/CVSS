from torch.utils.data import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from data.dataset import CVSS_train_dataset, CVSS_test_dataset
from sklearn.model_selection import train_test_split
from prefetch_generator import BackgroundGenerator
import torch
import torch.distributed as dist
class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def build_train_loader(config):
    series_ids_train = subfiles(config.DATASET.PROPRECESS_PATH, join=False, suffix='npz')
    
    if config.DATASET.WITH_VAL:
        series_ids = series_ids_train
        # series_ids_train, series_ids_val = train_test_split(series_ids_train, test_size=config.DATASET.VAL_SPLIT,random_state=0,shuffle =False)
        series_ids_val = series_ids[:int(config.DATASET.VAL_SPLIT*len(series_ids_train))]
        series_ids_train = series_ids[int(config.DATASET.VAL_SPLIT*len(series_ids_train)):]
        val_dataset = CVSS_train_dataset(config,series_ids_val, is_val=True)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset) if config.DIS else None
        val_loader = DataLoaderX(
            dataset=val_dataset,
            sampler=val_sampler ,
            batch_size = config.DATALOADER.BATCH_SIZE, 
            num_workers=config.DATALOADER.NUM_WORKERS,
            pin_memory= config.DATALOADER.PIN_MEMORY, 
            shuffle=False,
            drop_last=False
        )
    else:
        val_loader = None
    
    
    train_dataset = CVSS_test_dataset(config,series_ids_train, is_val=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size = config.DATALOADER.BATCH_SIZE, 
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory= config.DATALOADER.PIN_MEMORY, 
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    return train_loader,val_loader



def build_test_loader(config):
    series_ids_train = subfiles(config.DATASET.PROPRECESS_PATH, join=False, suffix='npz')
    

    
    train_dataset = CVSS_test_dataset(config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,shuffle=True) if config.DIS else None
    train_loader = DataLoaderX(
        train_dataset,
        sampler=train_sampler,
        batch_size = config.DATALOADER.BATCH_SIZE, 
        num_workers=config.DATALOADER.NUM_WORKERS,
        pin_memory= config.DATALOADER.PIN_MEMORY, 
        shuffle=True if train_sampler is None else False,
        drop_last=True
    )
    return train_loader,val_loader




