import glob
import os
import threading
import time
from pathlib import Path

import cv2

from utils import process_data

data_dir = Path('/media/linux/VOID/code/data/DoubleBag/Brutto')
one_bag = Path('/media/linux/VOID/code/data/DoubleBag/extra_data/1bag')
two_bags = Path('/media/linux/VOID/code/data/DoubleBag/extra_data/2bags')
eval_data = Path('/media/linux/VOID/code/data/DoubleBag/eval_data/2bag_detect_1')
window_name = "1: onebag, 2: twobags"


class SortingMachine(object):
    def __init__(self):
        self.exit = False
        self.file_to_img = dict()
        self.labelled_data = []
        for file in one_bag.glob('*.jpg'):
            self.labelled_data.append(str(file.name))

        for file in two_bags.glob('*.jpg'):
            self.labelled_data.append(str(file.name))

        for file in eval_data.glob('*.jpg'):
            self.labelled_data.append(str(file.name))

        self.dir_is_done = False

    def chunker_list(self, seq, size):
        return (seq[i::size] for i in range(size))

    def touch(self, path):
        with open(path, 'a'):
            os.utime(path, None)

    def preload_img(self, thread_nr, l):
        for file in l:
            if self.exit:
                break
            file_path = Path(file)
            if file_path.name in self.labelled_data:
                continue
            filename = file_path.name
            img = cv2.imread(str(file_path))
            cropped = process_data.crop_to_size(img)
            squared = process_data.pad_to_square(cropped)
            self.file_to_img[file_path.name] = squared

    def iterate_dirs(self):
        for obj in data_dir.glob("*"):
            if obj.is_dir():
                self.iterate_images(obj)
            while not self.dir_is_done:
                debug = 0
            self.dir_is_done = False

    def iterate_images(self, main_path):
        threads = 2
        l = [str(img_path) for img_path in main_path.glob("*.jpg")]
        seq = list(self.chunker_list(l, threads))
        for x in range(0, threads):
            threading.Thread(target=self.preload_img,
                             args=(x, seq[x])).start()
        for image in main_path.glob("*.jpg"):
            if self.exit:
                break
            if image.name in self.labelled_data:
                continue
            loading = True
            oldepoch = time.time()
            while loading:
                if image.name in self.file_to_img:
                    loading = False
                elif time.time() - oldepoch >= 1:
                    oldepoch = time.time()
                    print("loading file:", image.name)
            show = self.file_to_img[image.name]
            finish = False
            while not finish:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.imshow(window_name, show)
                cv2.moveWindow(window_name, 150 + 1 * 300, 40)

                k = cv2.waitKey(0)
                if k == ord('1'):
                    cv2.imwrite(str(Path(one_bag, image.name)), show)
                    # cv2.destroyAllWindows()
                    break
                elif k == ord('2'):
                    cv2.imwrite(str(Path(two_bags, image.name)), show)
                    # cv2.destroyAllWindows()
                    break
                elif k == ord('s'):
                    self.exit = True
                    break
                elif k == ord('n'):
                    break
            self.file_to_img[image.name] = None
        self.dir_is_done = True


if __name__ == '__main__':
    SortingMachine().iterate_dirs()
