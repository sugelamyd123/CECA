from torch.utils.data import Dataset
import logging
import torch
from utils.iotools import read_image
from utils.simple_tokenizer import SimpleTokenizer
from prettytable import PrettyTable
import random
import numpy as np
import os,json
from PIL import Image
from torchvision.transforms.functional import to_pil_image
from .chat_io_utils import (
    _norm_key, _read_text_file, _derive_chat_candidates_from_image
)
from typing import Dict

_CHAT_JSON_PATHS: Dict[str, str] = {
    "CUHK-PEDES": "/data01/zh_2023/text-image_retrival/CUHK-PEDES/CUHK-PEDES_CECA.json",
    "ICFG-PEDES": "/data01/zh_2023/text-image_retrival/ICFG-PEDES/ICFG-PEDES_CECA.json",
    "RSTPReid":   "/data01/zh_2023/text-image_retrival/RSTPReid/RSTP_CECA.json",
}

_JSON_INDEX_CACHE: Dict[str, Dict[str, str]] = {}  # key=dataset_root -> {key->summary}

def _dataset_name_from_img(p: str) -> str:
    p = _norm_key(p)
    for name in _CHAT_JSON_PATHS.keys():
        if name in p:
            return name
    return ""

def _dataset_root_from_img(p: str) -> str:
    p = _norm_key(p).split("/")
    if "imgs" in p:
        i = p.index("imgs")
        return "/".join(p[:i])       # /data/.../CUHK-PEDES
    return os.path.dirname(os.path.dirname(_norm_key("/".join(p))))

def _rel_from_imgs(p: str) -> str:
    p = _norm_key(p)
    return p.split("imgs/", 1)[1] if "imgs/" in p else os.path.basename(p)

def _pick_summary_from_dialogue(dlg) -> str:

    def clean(x):
        return x.strip() if isinstance(x, str) else ""


    if isinstance(dlg, dict):
        s = clean(dlg.get("summary", "")) or clean(dlg.get("text", ""))
        if s:
            return s

        a = clean(dlg.get("answer", ""))
        if a:
            return a
        q = clean(dlg.get("question", ""))
        if q or a:
            return (q + " " + a).strip()
        return ""


    if isinstance(dlg, list):

        if len(dlg) >= 5:
            s5 = _pick_summary_from_dialogue(dlg[4])
            if s5:
                return s5

        for it in reversed(dlg):
            s = _pick_summary_from_dialogue(it)
            if s:
                return s
        return ""


    return ""

def _get_summary_from_json_fallback(img_path: str) -> str:

    ds_name = _dataset_name_from_img(img_path)
    if not ds_name:
        return ""

    ds_root = _dataset_root_from_img(img_path)
    root_key = _norm_key(ds_root)
    rel = _norm_key(_rel_from_imgs(img_path))
    base = os.path.basename(rel)


    if root_key in _JSON_INDEX_CACHE:
        idx = _JSON_INDEX_CACHE[root_key]
        return idx.get(rel, "") or idx.get(base, "")

    json_path = _CHAT_JSON_PATHS.get(ds_name, "")
    if not json_path or not os.path.isfile(json_path):
        _JSON_INDEX_CACHE[root_key] = {}
        return ""

 
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        _JSON_INDEX_CACHE[root_key] = {}
        return ""

    index: Dict[str, str] = {}
    if isinstance(data, list):
        for it in data:
            if not isinstance(it, dict): 
                continue
            fp = it.get("file_path") or it.get("img_path") or it.get("image_path") or it.get("path") or ""
            dlg = it.get("dialogue", [])
            s = _pick_summary_from_dialogue(dlg)
            if not s:
                continue
            k1 = _norm_key(fp)
            k2 = os.path.basename(k1)
            if k1: index[k1] = s
            if k2: index[k2] = s
    elif isinstance(data, dict):
        for fp, dlg in data.items():
            k1 = _norm_key(fp)
            s = _pick_summary_from_dialogue(dlg)
            if not s:
                continue
            k2 = os.path.basename(k1)
            if k1: index[k1] = s
            if k2: index[k2] = s

    _JSON_INDEX_CACHE[root_key] = index
    return index.get(rel, "") or index.get(base, "")

def inject_noisy_correspondence(dataset, noisy_rate, noisy_file =None):
    logger = logging.getLogger("RDE.dataset")
    nums = len(dataset)
    dataset_copy = dataset.copy()
    captions  = [i[3] for i in dataset_copy]
    images    = [i[2] for i in dataset_copy]
    image_ids = [i[1] for i in dataset_copy]
    pids      = [i[0] for i in dataset_copy]

    noisy_inx = np.arange(nums)
    if noisy_rate > 0:
        print(noisy_file)
        random.seed(123)
        if os.path.exists(noisy_file):
            logger.info('=> Load noisy index from {}'.format(noisy_file))
            noisy_inx = np.load(noisy_file)
        else:
            inx = np.arange(nums)
            np.random.shuffle(inx)
            c_noisy_inx = inx[0: int(noisy_rate * nums)]
            shuffle_noisy_inx = np.array(c_noisy_inx)
            np.random.shuffle(shuffle_noisy_inx)
            noisy_inx[c_noisy_inx] = shuffle_noisy_inx
            np.save(noisy_file, noisy_inx)

    real_correspondeces = []
    for i in range(nums):
        if noisy_inx[i]== i:
            real_correspondeces.append(1)
        else:
            real_correspondeces.append(0)
        # pid, real_pid, image_id, image_path, text
        tmp = (pids[i],image_ids[i],images[i],captions[noisy_inx[i]])
        dataset[i] = tmp
    logger.info(real_correspondeces[0:10])
    logger.info('=>Noisy rate: {},  clean pairs: {}, noisy pairs: {}, total pairs: {}'.format(noisy_rate, np.sum(real_correspondeces),nums-np.sum(real_correspondeces), nums))

    return dataset, np.array(real_correspondeces)

class BaseDataset(object):
    """
    Base class of text to image reid dataset
    """
    logger = logging.getLogger("RDE.dataset")

    def show_dataset_info(self):
        num_train_pids, num_train_imgs, num_train_captions = len(
            self.train_id_container), len(self.train_annos), len(self.train)
        num_test_pids, num_test_imgs, num_test_captions = len(
            self.test_id_container), len(self.test_annos), len(
                self.test['captions'])
        num_val_pids, num_val_imgs, num_val_captions = len(
            self.val_id_container), len(self.val_annos), len(
                self.val['captions'])

        # TODO use prettytable print comand line table

        self.logger.info(f"{self.__class__.__name__} Dataset statistics:")
        table = PrettyTable(['subset', 'ids', 'images', 'captions'])
        table.add_row(
            ['train', num_train_pids, num_train_imgs, num_train_captions])
        table.add_row(
            ['test', num_test_pids, num_test_imgs, num_test_captions])
        table.add_row(['val', num_val_pids, num_val_imgs, num_val_captions])
        self.logger.info('\n' + str(table))


def tokenize(caption: str, tokenizer, text_length=77, truncate=True) -> torch.LongTensor:
    sot_token = tokenizer.encoder["<|startoftext|>"]
    eot_token = tokenizer.encoder["<|endoftext|>"]
    tokens = [sot_token] + tokenizer.encode(caption) + [eot_token]

    result = torch.zeros(text_length, dtype=torch.long)
    if len(tokens) > text_length:
        if truncate:
            tokens = tokens[:text_length]
            tokens[-1] = eot_token
        else:
            raise RuntimeError(
                f"Input {caption} is too long for context length {text_length}"
            )
    result[:len(tokens)] = torch.tensor(tokens)
    return result
 


class ImageDataset(Dataset):
  
    def __init__(self,
                 image_pids,
                 img_paths,
                 transform=None,
                 args=None,
                 chat_index=None,        
                 chat_length: int = 77,
                 truncate: bool = True):
        assert len(image_pids) == len(img_paths)
        self.image_pids = image_pids
        self.img_paths = img_paths
        self.transform = transform

        self.tokenizer = SimpleTokenizer()
        self.chat_length = chat_length
        self.truncate = truncate

       
        self.require_chat = bool(getattr(args, "require_chat", False)) if args else False
       
        self.chat_dir = getattr(args, "chat_dir", "chats") if args else "chats"
        exts = getattr(args, "chat_exts", ".txt,.json") if args else ".txt,.json"
        self.chat_exts = [s.strip() for s in exts.split(",") if s.strip()]
        self._cache = {}

    def __len__(self): 
        return len(self.image_pids)

    def _read_and_transform(self, img_path):
        img = read_image(img_path)  
        if self.transform is None:
            return img
        if isinstance(img, torch.Tensor):
            try:
                return self.transform(img)
            except Exception:
                return self.transform(to_pil_image(img))
        if isinstance(img, Image.Image):
            try:
                return self.transform(img)
            except Exception:
                return self.transform(to_pil_image(torch.from_numpy(np.array(img)).permute(2,0,1)))
        return self.transform(img)

    def _load_chat_text(self, img_path: str) -> str:
        key = _norm_key(img_path)
        if key in self._cache:
            return self._cache[key]

       
        for cand in _derive_chat_candidates_from_image(key, self.chat_dir, self.chat_exts):
            if os.path.isfile(cand):
                text = _read_text_file(cand)
                if text:
                    self._cache[key] = text
                    return text

       
        text = _get_summary_from_json_fallback(key)
        if text:
            self._cache[key] = text
            return text

       
        self._cache[key] = ""
        return ""

    def __getitem__(self, index):
        pid = self.image_pids[index]
        img_path = self.img_paths[index]
        img = self._read_and_transform(img_path)

        chat_text = self._load_chat_text(img_path)
        if self.require_chat and not chat_text:
            raise KeyError(f"Chat not found for: {img_path}")

        summary_ids = tokenize(
            chat_text if chat_text else "",
            tokenizer=self.tokenizer,
            text_length=self.chat_length,
            truncate=self.truncate
        )
        return pid, img, summary_ids

class TextDataset(Dataset):
    def __init__(self,
                 caption_pids,
                 captions,
                 text_length: int = 77,
                 truncate: bool = True):
        self.caption_pids = caption_pids
        self.captions = captions
        self.text_length = text_length
        self.truncate = truncate
        self.tokenizer = SimpleTokenizer()
  
    def __len__(self):
        return len(self.caption_pids)

    def __getitem__(self, index):
        pid, caption = self.caption_pids[index], self.captions[index]
        caption = tokenize(caption, tokenizer=self.tokenizer, text_length=self.text_length, truncate=self.truncate)

        return pid, caption


class ImageTextDataset(Dataset):

    def __init__(self,
                 dataset, args,
                 transform=None,
                 text_length: int = 77,
                 truncate: bool = True,
                 chat_index=None):  
        self.dataset = dataset
        self.transform = transform
        self.text_length = text_length
        self.truncate = truncate

        self.txt_aug  = getattr(args, "txt_aug", False)
        self.img_aug  = getattr(args, "img_aug", False)
        self.chat_aug = getattr(args, "chat_aug", False)

        
        self.real_correspondences = np.ones(len(self.dataset), dtype=np.int32)

        self.tokenizer = SimpleTokenizer()
        self.chat_length = getattr(args, "chat_length", text_length)
        self.require_chat = bool(getattr(args, "require_chat", False))

       
        self.chat_dir = getattr(args, "chat_dir", "chats")
        exts = getattr(args, "chat_exts", ".txt,.json")
        self.chat_exts = [s.strip() for s in exts.split(",") if s.strip()]
        self._cache = {}

    def __len__(self): 
        return len(self.dataset)

    def _read_and_transform(self, img_path):
        img = read_image(img_path)
        if self.transform is None:
            return img
        if isinstance(img, torch.Tensor):
            try:
                return self.transform(img)
            except Exception:
                return self.transform(to_pil_image(img))
        if isinstance(img, Image.Image):
            try:
                return self.transform(img)
            except Exception:
                return self.transform(to_pil_image(torch.from_numpy(np.array(img)).permute(2,0,1)))
        return self.transform(img)

    def _load_chat_text(self, img_path: str) -> str:
        key = _norm_key(img_path)
        if key in self._cache:
            return self._cache[key]

        
        for cand in _derive_chat_candidates_from_image(key, self.chat_dir, self.chat_exts):
            if os.path.isfile(cand):
                text = _read_text_file(cand)
                if text:
                    self._cache[key] = text
                    return text

      
        text = _get_summary_from_json_fallback(key)
        if text:
            self._cache[key] = text
            return text

        self._cache[key] = ""
        return ""

    def __getitem__(self, index):
        pid, image_id, img_path, caption = self.dataset[index]
        img = self._read_and_transform(img_path)

        caption_ids = tokenize(
            caption, tokenizer=self.tokenizer,
            text_length=self.text_length, truncate=self.truncate
        )
        if self.txt_aug:
            caption_ids = self.txt_data_aug(caption_ids.cpu().numpy())

        chat_text = self._load_chat_text(img_path)
        if self.require_chat and not chat_text:
            raise KeyError(f"Chat not found for: {img_path}")

        chat_ids = tokenize(
            chat_text if chat_text else "",
            tokenizer=self.tokenizer,
            text_length=self.chat_length,
            truncate=True
        )
        if self.chat_aug:
            chat_ids = self.txt_data_aug(chat_ids.cpu().numpy())

        return {
            'pids': pid,
            'image_ids': image_id,
            'images': img,
            'caption_ids': caption_ids,
            'chat_ids': chat_ids,
            'index': index,
        }

    def txt_data_aug(self, tokens):
        mask = self.tokenizer.encoder["<|mask|>"]
        vocab_hi = len(self.tokenizer.encoder) - 3  #
        token_range = list(range(1, vocab_hi))
        new_tokens = np.zeros_like(tokens)
        aug_tokens = []
        for i, token in enumerate(tokens):
            if 0 < token < vocab_hi:
                prob = random.random()
                if prob < 0.20:
                    prob /= 0.20
                    if prob < 0.6:   aug_tokens.append(mask)
                    elif prob < 0.8: aug_tokens.append(random.choice(token_range))
                    else:            pass
                else:
                    aug_tokens.append(tokens[i])
            else:
                aug_tokens.append(tokens[i])
        new_tokens[0:len(aug_tokens)] = np.array(aug_tokens)
        return torch.tensor(new_tokens, dtype=torch.long)