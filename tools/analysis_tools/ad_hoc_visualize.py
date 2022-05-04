import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser("ad hoc visualize")
    parser.add_argument("--folder", type=str ,default="visualization/diff")
    args = parser.parse_args()
    return args

def process_checkpoint(checkpoint_path):
    file_npz = os.path.join(checkpoint_path, "features", "svd.npz.npy")
    svd_log = np.load(file_npz)
    return svd_log

def process_checkpoints_list(checkpoints_list, file_name, checkpoint_folder):
    for checkpoint in checkpoints_list:
        # do smth
        svd_log = process_checkpoint(checkpoint)
        plt.plot(svd_log, label=os.path.basename(checkpoint))
        pass
    plt.legend()
    plt.savefig(os.path.join(checkpoint_folder, file_name))
    plt.close()


if __name__ == '__main__':
    args = parse_args()
    checkpoints_path= args.folder
    # checkpoints_list = list(glob(os.path.join(checkpoints_path, "*")))
    # print(checkpoints_list)
    checkpoints_list1 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "neg"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list1, "implicit.png", checkpoints_path)

    checkpoints_list2 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list2, "simdis.png", checkpoints_path)

    checkpoints_list3 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "pos"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list3, "explicit.png", checkpoints_path)

    checkpoints_list4 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "neg"),
        os.path.join(checkpoints_path, "pos"),
        os.path.join(checkpoints_path, "poswneg")
    ]
    process_checkpoints_list(checkpoints_list4, "impvsexp.png", checkpoints_path)

    checkpoints_list5 = [
        os.path.join(checkpoints_path, "simsiam"),
        os.path.join(checkpoints_path, "poswneg"),
        os.path.join(checkpoints_path, "simdis")
    ]
    process_checkpoints_list(checkpoints_list5, "simdisvsimpexp.png", checkpoints_path)

    checkpoints_list6 = [
        os.path.join(checkpoints_path, 'simsiam_r50'),
        os.path.join(checkpoints_path, 'simsiam'),
        os.path.join(checkpoints_path, 'simdis')
    ]
    process_checkpoints_list(checkpoints_list6, "teacher.png", checkpoints_path)
    pass