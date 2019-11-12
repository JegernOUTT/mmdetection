## Be ready
- First of all you need to build mmdet with using of its own **INSTALL.md**
- Make sure that your **mmdet** is fully workable (try to train some simple detector, r18_retinanet or etc)

## Data preparation (**data_loader.py** creation)
- Think about your training / val / test sets, classes, datasets, and other "data" stuff
- Use detector_utils/data_loader to load data locally (you do not have to copy all you pic data on server but it is the preferred way)
- Make runnable script with data loading configs. See and use **mmdetection/tools/dssl_data_loader.py**

## Experiments
- Create your own branch with "speaking" name
- The prefer way is to use CSV for dumped annotation file and other "low weight" artifacts
- Make commit for every training run with good describing of last experiments changing
- Use **wandb** for logging and traking experiments  (see **mmdetection/configs/wandb_example**)
- Use separate projects in **wandb** to log and dump all your training artifacts
