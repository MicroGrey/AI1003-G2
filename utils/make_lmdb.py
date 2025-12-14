import os
import glob
from basicsr.utils.lmdb_util import make_lmdb_from_imgs

def prepare_lmdb():
    # 0. 准备存放目录
    save_folder = './my_lmdbs'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # =========================================================
    # 1. 制作 HR LMDB (保持不变)
    # =========================================================
    print("开始制作 HR LMDB ...")
    hr_input_folder = 'datasets/DF2K/HR'
    
    full_path_list = sorted(glob.glob(os.path.join(hr_input_folder, '*')))
    # 只保留文件名
    img_path_list = [os.path.basename(p) for p in full_path_list]
    # 生成 keys: '0001.png' -> '0001'
    keys = [os.path.splitext(v)[0] for v in img_path_list]

    lmd_path = os.path.join(save_folder, 'DF2K_train_HR.lmdb')
    
    make_lmdb_from_imgs(
        data_path=hr_input_folder,
        lmdb_path=lmd_path,
        img_path_list=img_path_list,
        keys=keys,
        batch=2000,
        compress_level=1,
        multiprocessing_read=True,
        n_thread=40,
        map_size=None
    )

    # =========================================================
    # 2. 制作 LR LMDB (重点修改这里)
    # =========================================================
    print("开始制作 LR LMDB ...")
    lr_input_folder = 'datasets/DF2K/LR_bicubic/X2'
    
    full_path_list = sorted(glob.glob(os.path.join(lr_input_folder, '*')))
    img_path_list = [os.path.basename(p) for p in full_path_list]
    
    # [关键修复]：如果文件名带 'x2'，把 Key 里的 x2 去掉
    # 例如 '0001x2.png' -> key 变成 '0001'，而不是 '0001x2'
    keys = []
    for v in img_path_list:
        key = os.path.splitext(v)[0]
        if key.endswith('x2'):
            key = key[:-2] # 去掉末尾的 x2
        keys.append(key)
    
    lmd_path = os.path.join(save_folder, 'DF2K_train_LR_bicubic_X2.lmdb')
    
    make_lmdb_from_imgs(
        data_path=lr_input_folder,
        lmdb_path=lmd_path,
        img_path_list=img_path_list,
        keys=keys, # 现在这里的 keys 和 HR 是一模一样的了
        batch=2000,
        compress_level=1,
        multiprocessing_read=True,
        n_thread=40,
        map_size=None
    )

if __name__ == '__main__':
    prepare_lmdb()