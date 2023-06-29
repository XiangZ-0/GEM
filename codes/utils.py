import os
from pathlib2 import Path

def load_model_path(root=None, stage=None, cycle=None):
    def sort_by_epoch(path):
        name = path.stem
        epoch=int(name.split('-')[1].split('=')[1])
        return epoch
    
    def generate_root():
        if root is not None:
            return root
        elif stage is not None:
            return str(Path('lightning_logs', stage, 'checkpoints'))
        else:
            return str(Path('lightning_logs', f'stage_{cycle}', 'checkpoints'))

    if root==stage==cycle==None:
        return None

    if stage == 1:
        cycle = 0
    prefix = 'best-stage=%01d-cycle=%02d'%(stage, cycle)
    files=[i for i in list(Path(root).iterdir()) if i.stem.startswith(prefix)]
    files.sort(key=sort_by_epoch, reverse=True)
    res = str(files[0])

    return res

def load_model_path_by_args(args):
    return load_model_path(root=args.load_dir, stage=args.load_stage, cycle=args.load_cycle)
