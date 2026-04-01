import os, json
from typing import Optional, List, Dict

def _norm_key(p):
    if not p:
        return ""
    return os.path.normpath(p).replace("\\", "/")

def _read_text_file(path: str) -> str:
    try:
        ext = os.path.splitext(path)[1].lower()
        if ext in ("", ".txt"):
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
        elif ext == ".json":
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
               
                return data.get("summary") or data.get("text") or json.dumps(data, ensure_ascii=False)
            return json.dumps(data, ensure_ascii=False)
        else:
            with open(path, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        return ""

def _derive_chat_candidates_from_image(img_path: str,
                                       chat_dir: str = "chats",
                                       chat_exts: List[str] = None) -> List[str]:
  
    chat_exts = chat_exts or [".txt", ".json"]
    p = _norm_key(img_path)
    parts = p.split("/")
    candidates = []

    if "imgs" in parts:
        idx = parts.index("imgs")
        root = "/".join(parts[:idx])            # imgs 
        rel_from_imgs = "/".join(parts[idx+1:]) # imgs 
    else:
        root = os.path.dirname(p)
        rel_from_imgs = os.path.basename(p)

    stem = os.path.splitext(os.path.basename(p))[0]
    img_dir = os.path.dirname(p)

    # 1) <root>/<chat_dir>/<rel_without_ext>.<ext>
    rel_wo_ext = os.path.splitext(rel_from_imgs)[0]
    if root:
        for ext in chat_exts:
            candidates.append(_norm_key(os.path.join(root, chat_dir, rel_wo_ext + ext)))

    # 2) same
    for ext in chat_exts:
        candidates.append(_norm_key(os.path.join(img_dir, stem + ext)))

    # 3) <root>/<stem>.<ext>
    if root:
        for ext in chat_exts:
            candidates.append(_norm_key(os.path.join(root, stem + ext)))

 
    out, seen = [], set()
    for c in candidates:
        if c not in seen:
            seen.add(c); out.append(c)
    return out
def read_text_from_file(p: str) -> str:
    try:
        ext = os.path.splitext(p)[1].lower()
        if ext in ('.txt', ''):
            with open(p, 'r', encoding='utf-8') as f:
                return f.read().strip()
        elif ext == '.json':
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data.get('summary') or data.get('text') or json.dumps(data, ensure_ascii=False)
            return json.dumps(data, ensure_ascii=False)
        else:
            with open(p, 'r', encoding='utf-8') as f:
                return f.read().strip()
    except Exception:
        return ""

def candidate_json_keys(img_path: str, dataset_root: Optional[str]) -> List[str]:
    p = _norm_key(img_path)
    keys: List[str] = []
    if dataset_root:
        r = _norm_key(dataset_root)
        if p.startswith(r + '/'):
            keys.append(p[len(r) + 1:])         
    if 'imgs/' in p:
        keys.append(p.split('imgs/', 1)[1])       
    keys.append(os.path.basename(p))                 
    keys.append(p)                                   

    out: List[str] = []
    seen = set()
    for k in keys:
        k = _norm_key(k)
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

import os, json
from typing import Any, Dict

def build_chat_index(chat_json_map: Any) -> Dict[str, str]:
    
    norm_map: Dict[str, str] = {}

    def _add_one(path: str, alias: str = "_"):
        if path and isinstance(path, str):
            norm_map[alias] = path

    if isinstance(chat_json_map, dict):
        # {dataset_name: json_path}
        for k, v in chat_json_map.items():
            if isinstance(v, str):
                norm_map[str(k)] = v
    elif isinstance(chat_json_map, (list, tuple)):
        # [json_path1, json_path2, ...]
        for i, v in enumerate(chat_json_map):
            if isinstance(v, str):
                norm_map[str(i)] = v
    elif isinstance(chat_json_map, str):
       
        if os.path.isfile(chat_json_map):
            _add_one(chat_json_map, "_")
        else:
            try:
                parsed = json.loads(chat_json_map)
                if isinstance(parsed, dict):
                    for k, v in parsed.items():
                        if isinstance(v, str):
                            norm_map[str(k)] = v
                elif isinstance(parsed, (list, tuple)):
                    for i, v in enumerate(parsed):
                        if isinstance(v, str):
                            norm_map[str(i)] = v
                else:
                    
                    _add_one(chat_json_map, "_")
            except Exception:
               
                _add_one(chat_json_map, "_")
    else:
       
        return {}


    idx: Dict[str, str] = {}
    if not norm_map:
        return idx

    for _, json_path in norm_map.items():
        if not os.path.isfile(json_path):
            print(f"[WARN] chat json not found: {json_path}")
            continue
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[WARN] failed to load json: {json_path} ({e})")
            continue

       
        if isinstance(data, list):
            items = [( (it.get("file_path","") if isinstance(it, dict) else ""), 
                       (it.get("dialogue", []) if isinstance(it, dict) else []) ) for it in data]
        elif isinstance(data, dict):
            # 
            items = list(data.items())
        else:
            continue

        def pick_summary(dlg):
            #
            if not dlg:
                return ""
            if isinstance(dlg, dict):
                
                return dlg.get("summary","") or dlg.get("text","")
            if not isinstance(dlg, list):
                return ""
            if len(dlg) >= 5 and isinstance(dlg[4], dict):
                return dlg[4].get("summary","") or dlg[4].get("text","")
            last = dlg[-1]
            return (last.get("summary","") or last.get("text","")) if isinstance(last, dict) else ""

        for fp, dlg in items:
            s = pick_summary(dlg)
            if not s:
                continue
            k1 = _norm_key(fp)
            k2 = os.path.basename(k1) if k1 else ""
            if k1:
                idx[k1] = s
            if k2:
                idx[k2] = s

    return idx
def detect_dataset_name(img_path: str, name_keys):
    p = _norm_key(img_path)
    for name in name_keys:
        if _norm_key(name) in p:
            return name
    return None

def lookup_chat_json(img_path: str, chat_index: Dict[str, str], dataset_root: Optional[str]) -> str:
    for k in candidate_json_keys(img_path, dataset_root):
        if k in chat_index:
            return chat_index[k]
    return ""

def derive_chat_file_candidates(
    img_path: str,
    dataset_root: Optional[str],
    chat_dirs: Optional[List[str]] = None,
    chat_exts: Optional[List[str]] = None
) -> List[str]:
  
    chat_dirs = chat_dirs or ['chats', 'captions', 'chat']
    chat_exts = chat_exts or ['.txt', '.json']

    p = _norm_key(img_path)
    root = _norm_key(dataset_root) if dataset_root else None
    rel_from_imgs = None

    if 'imgs/' in p:
        rel_from_imgs = p.split('imgs/', 1)[1]
    elif root and p.startswith(root + '/'):
        rel_from_imgs = p[len(root) + 1:]
    else:
        rel_from_imgs = os.path.basename(p)

    stem = os.path.splitext(os.path.basename(p))[0]
    img_dir = os.path.dirname(p)

    candidates: List[str] = []

    # 1) <root>/<chat_dir>/<rel_from_imgs with new ext>
    if root:
        for d in chat_dirs:
            for ext in chat_exts:
                candidates.append(_norm_key(os.path.join(root, d, os.path.splitext(rel_from_imgs)[0] + ext)))

    # 2) same
    for ext in chat_exts:
        candidates.append(_norm_key(os.path.join(img_dir, stem + ext)))

    # 3) <root>/<stem>.<ext>
    if root:
        for ext in chat_exts:
            candidates.append(_norm_key(os.path.join(root, stem + ext)))


    out: List[str] = []
    seen = set()
    for c in candidates:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out
