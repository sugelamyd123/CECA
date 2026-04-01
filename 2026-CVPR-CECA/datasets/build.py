import logging
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from datasets.sampler import RandomIdentitySampler
from datasets.sampler_ddp import RandomIdentitySampler_DDP

from torch.utils.data import BatchSampler

from utils.comm import get_world_size

from .bases import ImageDataset, TextDataset, ImageTextDataset
from .cuhkpedes import CUHKPEDES
from .icfgpedes import ICFGPEDES
from .rstpreid import RSTPReid


from .chat_io_utils import build_chat_index

__factory = {'CUHK-PEDES': CUHKPEDES, 'ICFG-PEDES': ICFGPEDES, 'RSTPReid': RSTPReid}


def build_transforms(img_size=(384, 128), aug=False, is_train=True):
    height, width = img_size
    mean = [0.48145466, 0.4578275, 0.40821073]
    std = [0.26862954, 0.26130258, 0.27577711]

    if not is_train:
        return T.Compose([
            T.Resize((height, width)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])

    if aug:
        
        return T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.Pad(10),
            T.RandomCrop((height, width)),
            T.ToTensor(),
            T.RandomErasing(scale=(0.02, 0.4), value=mean),
            T.Normalize(mean=mean, std=std),
        ])
    else:
        return T.Compose([
            T.Resize((height, width)),
            T.RandomHorizontalFlip(0.5),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])


def collate(batch):
    
    keys = set([key for b in batch for key in b.keys()])
    dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

    batch_tensor_dict = {}
    for k, v in dict_batch.items():
        first = v[0]
        if isinstance(first, int):
            batch_tensor_dict[k] = torch.tensor(v)
        elif torch.is_tensor(first):
            batch_tensor_dict[k] = torch.stack(v)
        else:
           
            raise TypeError(f"Unexpect data type: {type(v[0])} in a batch.")
    return batch_tensor_dict


def build_dataloader(args, tranforms=None):
    logger = logging.getLogger("IRRA.dataset")

    num_workers = args.num_workers
    dataset = __factory[args.dataset_name](root=args.root_dir)
    num_classes = len(dataset.train_id_container)

    
    shared_chat_index = build_chat_index(getattr(args, "chat_json_map", {}) or {})

    if args.training:
        train_transforms = build_transforms(img_size=args.img_size,
                                            aug=args.img_aug,
                                            is_train=True)
        val_transforms = build_transforms(img_size=args.img_size,
                                          is_train=False)

       
        train_set = ImageTextDataset(dataset.train, args,
                                     train_transforms,
                                     text_length=args.text_length,
                                     chat_index=shared_chat_index)

        if args.sampler == 'identity':
            if args.distributed:
                logger.info('using ddp random identity sampler')
                logger.info('DISTRIBUTED TRAIN START')

                mini_batch_size = args.batch_size // get_world_size()
                data_sampler = RandomIdentitySampler_DDP(
                    dataset.train, args.batch_size, args.num_instance)
                batch_sampler = BatchSampler(data_sampler, mini_batch_size, drop_last=True)

              
                train_loader = DataLoader(train_set,
                                          batch_sampler=batch_sampler,
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          pin_memory=True,
                                          persistent_workers=(num_workers > 0))
            else:
                logger.info(
                    f'using random identity sampler: batch_size: {args.batch_size}, id: {args.batch_size // args.num_instance}, instance: {args.num_instance}'
                )
                train_loader = DataLoader(train_set,
                                          batch_size=args.batch_size,
                                          sampler=RandomIdentitySampler(
                                              dataset.train, args.batch_size,
                                              args.num_instance),
                                          num_workers=num_workers,
                                          collate_fn=collate,
                                          pin_memory=True,
                                          persistent_workers=(num_workers > 0))
        elif args.sampler == 'random':
            logger.info('using random sampler')
            train_loader = DataLoader(train_set,
                                      batch_size=args.batch_size,
                                      shuffle=True,
                                      num_workers=num_workers,
                                      collate_fn=collate,
                                      pin_memory=True,
                                      persistent_workers=(num_workers > 0))
        else:
            logger.error('unsupported sampler! expected identity or random but got {}'.format(args.sampler))
            raise ValueError("unsupported sampler")

     
        ds = dataset.val if args.val_dataset == 'val' else dataset.test
        val_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                   val_transforms,
                                   args=args,
                                   chat_index=shared_chat_index,
                                   chat_length=args.text_length)
        val_txt_set = TextDataset(ds['caption_pids'],
                                  ds['captions'],
                                  text_length=args.text_length)

        val_img_loader = DataLoader(val_img_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    persistent_workers=(num_workers > 0))
        val_txt_loader = DataLoader(val_txt_set,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=num_workers,
                                    pin_memory=True,
                                    persistent_workers=(num_workers > 0))

        return train_loader, val_img_loader, val_txt_loader, num_classes

    else:

        test_transforms = tranforms if tranforms else build_transforms(img_size=args.img_size, is_train=False)

        ds = dataset.test

        test_img_set = ImageDataset(ds['image_pids'], ds['img_paths'],
                                    test_transforms,
                                    args=args,
                                    chat_index=shared_chat_index,
                                    chat_length=args.text_length)
        test_txt_set = TextDataset(ds['caption_pids'],
                                   ds['captions'],
                                   text_length=args.text_length)

        test_img_loader = DataLoader(test_img_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     persistent_workers=(num_workers > 0))
        test_txt_loader = DataLoader(test_txt_set,
                                     batch_size=args.test_batch_size,
                                     shuffle=False,
                                     num_workers=num_workers,
                                     pin_memory=True,
                                     persistent_workers=(num_workers > 0))
        return test_img_loader, test_txt_loader, num_classes
