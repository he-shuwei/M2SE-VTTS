
import importlib
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from m2se_vtts.utils.hparams import set_hparams, hparams

def _import_task_cls(dotted_path: str):
    module_path, cls_name = dotted_path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)

if __name__ == '__main__':
    set_hparams()
    task_cls_str = hparams.get(
        'task_cls',
        'm2se_vtts.tasks.m2se_vtts_task.M2SETask',
    )
    task_cls = _import_task_cls(task_cls_str)
    task_cls.start()
