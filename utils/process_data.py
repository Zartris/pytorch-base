import copy
import math
import random
from pathlib import Path
import cv2
import threading

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def process_all(ip, op, width, height, filter, bilateral):
    if not op.exists():
        op.mkdir(parents=True)
    for element in ip.glob("*"):
        if element.is_dir():
            threading.Thread(target=process_all,
                             args=(Path(element), Path(op, element.name), width, height, filter, bilateral)).start()
        elif element.suffix == ".jpg" or element.suffix == ".png":
            process_img(bilateral, element, filter, height, op, width)


def process_list(l, op, width, height, filter, bilateral):
    for img_path in l:
        process_img(bilateral, Path(img_path), filter, height, op, width)


def process_img(bilateral, img_path, filter, height, op, width):
    img = cv2.imread(str(img_path))
    # gray = gray_scale(img)
    square_img = pad_to_square(img)
    resized_img = resize(width, height, square_img)
    # smooth_img = blur(resized_img, filter, bilateral)
    cv2.imwrite(str(Path(op, img_path.name)), resized_img)
    print("Writing", str(Path(op, img_path.name)))


def pad_to_square(img):
    img_h, img_w = img.shape[:2]
    if img_h > img_w:
        border = img_h - img_w
        return cv2.copyMakeBorder(copy.copy(img), 0, 0, math.floor(border / 2), math.ceil(border / 2),
                                  cv2.BORDER_CONSTANT)
    else:
        border = img_w - img_h
        return cv2.copyMakeBorder(copy.copy(img), math.floor(border / 2), math.ceil(border / 2), 0, 0,
                                  cv2.BORDER_CONSTANT)


def gray_scale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize(width, height, img):
    return cv2.resize(img, (height, width))


def blur(img, filter, bilateral, d=10):
    if filter[0] == 0:
        return img

    result = copy.copy(img)
    if bilateral:
        one, two = filter
        cv2.bilateralFilter(src=img, dst=result, d=d,
                            sigmaColor=two, sigmaSpace=one)
    else:
        result = cv2.GaussianBlur(src=img,
                                  ksize=filter, sigmaX=0)
    return result


def background_removal(folder):
    height, width = (299, 299)
    filter = (150, 150)
    bil = True
    fgbg = cv2.createBackgroundSubtractorMOG2()
    for element in folder.glob("*.jpg"):
        img = cv2.imread(str(element), cv2.IMREAD_COLOR)
        resized = resize(width, height, img)
        gray = gray_scale(resized)
        blurred = blur(gray, (75, 75), True, d=7)
        laplacian = cv2.Laplacian(blurred, cv2.CV_8U)
        sobelx = cv2.Sobel(blurred, cv2.CV_16S, 1, 0, ksize=5)
        sobely = cv2.Sobel(blurred, cv2.CV_16S, 0, 1, ksize=5)
        sobelx = blur(sobelx, (9, 9), False)
        sobely = blur(sobely, (9, 9), False)

        abs_grad_x = cv2.convertScaleAbs(sobelx)
        abs_grad_y = cv2.convertScaleAbs(sobely)

        grad = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        # fgmask = fgbg.apply(blurred)
        # blurred = blur(resized, (75, 57), True, d=11)
        # blurred_float = blurred.astype(np.float32) / 255.0
        # edgeDetector = cv2.ximgproc.createStructuredEdgeDetection("cv_model/model.yml")
        # edges = edgeDetector.detectEdges(blurred_float) * 255.0

        cv2.imshow('res', sobelx)
        cv2.imshow('res1', sobely)
        cv2.imshow('res2', grad)
        cv2.imshow("blurred", laplacian)
        cv2.moveWindow('blurred', 400, 400)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def edgedetect(channel):
    sobelX = cv2.Sobel(channel, cv2.CV_16S, 1, 0, ksize=5)
    sobelY = cv2.Sobel(channel, cv2.CV_16S, 0, 1, ksize=5)
    sobel = np.hypot(sobelX, sobelY)

    sobel[sobel > 255] = 255  # Some values seem to go above 255. However RGB channels has to be within 0-255


def cluster_img(folder):
    height, width = (299, 299)
    filter = (150, 150)
    bil = True
    for element in folder.glob("*.jpg"):
        img = cv2.imread(str(element))
        gray = gray_scale(img)
        square_img = pad_to_square(gray)
        resized_img = resize(width, height, square_img)
        smooth_img = blur(resized_img, filter, bil)
        cv2.imshow('blurred', smooth_img)
        img = smooth_img
        Z = img.reshape((-1, 1))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        cv2.imshow('res2', res2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # img = plt.imread(img_in) / 255
    # blur(img, (25, 25), True)
    # plt.imshow(img)
    # pic_n = img.reshape(img.shape[0] * img.shape[1], img.shape[2])
    # kmeans = KMeans(n_clusters=4, random_state=0).fit(pic_n)
    # pic2show = kmeans.cluster_centers_[kmeans.labels_]
    # cluster_pic = pic2show.reshape(img.shape[0], img.shape[1], img.shape[2])
    # plt.imshow(cluster_pic)
    # plt.show()


def compute_ratio(width, height, img):
    img_h, img_w = img.shape[:2]


def chunker_list(seq, size):
    return (seq[i::size] for i in range(size))


def split_data(input_path, output_path, threads=4):
    rand_list = []
    for x in range(0, 250):
        while True:
            ran = random.randint(0, 1550)
            if ran not in rand_list:
                rand_list.append(ran)
                break

    bag1_path = Path(input_path, "1bag")
    bag2_path = Path(input_path, "2bags")
    train_path = create_folder(Path(output_path, "train"))
    val_path = create_folder(Path(output_path, "val"))
    create_folder(Path(train_path, "1bag"))
    create_folder(Path(train_path, "2bags"))
    create_folder(Path(val_path, "1bag"))
    create_folder(Path(val_path, "2bags"))
    new_list = [bag1_path, bag2_path]
    worker_job_list = []
    for x in range(threads):
        worker_job_list.append([])
    index = 0
    for folder in new_list:
        folder_name = folder.name
        counter = 0
        for element in folder.glob("*"):
            if element.suffix == ".jpg" or element.suffix == ".png":
                # img = cv2.imread(str(element))
                if counter in rand_list:
                    dest = Path(val_path, folder_name, element.name)
                else:
                    dest = Path(train_path, folder_name, element.name)
                worker_job_list[index % threads].append((str(element), str(dest)))
                # cv2.imwrite(str(dest), img)
                # print(str(dest))
                index += 1
                counter += 1
    for i in range(threads):
        threading.Thread(target=copy_to_dest,
                         args=(i, worker_job_list[i])).start()


def copy_to_dest(thread_nr, data_list):
    for data in data_list:
        ip, op = data
        cv2.imwrite(op, cv2.imread(ip))
        print(str(thread_nr), ":", op)


def reset_folder(folder):
    debug = 0


def create_folder(folder):
    if not folder.exists():
        folder.mkdir(parents=True)
    return folder


if __name__ == '__main__':
    input_path = Path("/media/linux/VOID/code/data/DoubleBag/trainingset_pt/")
    # input_path = Path("/media/linux/VOID/code/data/DoubleBag/eval_data/2bag_detect_1")
    output_path = Path("/media/linux/VOID/code/data/DoubleBag/trainingset_processed/")
    # output_path = Path("/media/linux/VOID/code/data/DoubleBag/trainingset_pt/")
    deligate = False
    split = False
    test = False
    test_process = True

    if split:
        split_data(input_path, output_path)

    if deligate:
        threading.Thread(target=process_all,
                         args=(input_path, Path(output_path, "resized_color_50"), 224, 224, (50, 50), True)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "blur_5_5"), 299, 299, (5, 5), False)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "blur_9_9"), 299, 299, (9, 9), False)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "blur_0_0"), 299, 299, (0, 0), False)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "bilateral_75_75"), 299, 299, (75, 75), True)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "bilateral_50_50"), 299, 299, (50, 50), True)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "bilateral_25_25"), 299, 299, (25, 25), True)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "bilateral_10_10"), 299, 299, (10, 10), True)).start()
        # threading.Thread(target=process_all,
        #                  args=(input_path, Path(output_path, "bilateral_150_150"), 299, 299, (150, 150), True)).start()
    if test:
        folder = Path(input_path, "train", "2bags")
        background_removal(folder)
    if test_process:
        input_path = Path("/media/linux/VOID/code/data/DoubleBag/eval_data/2bag_detect_1")
        output_path = Path("/media/linux/VOID/code/data/DoubleBag/eval_data/test_processed")
        if not output_path.exists():
            output_path.mkdir(parents=True)

        name = "eval_resized_color"
        if not Path(output_path, name).exists():
            Path(output_path, name).mkdir(parents=True)

        l = [str(img_path) for img_path in input_path.glob("*.jpg")]
        seq = list(chunker_list(seq=l, size=4))
        for i in range(4):
            threading.Thread(target=process_list,
                             args=(
                                 seq[i], Path(output_path, name), 224, 224, (50, 50),
                                 True)).start()


def crop_to_size(img):
    h, w = img.shape[:2]
    crop_w = (w - 820) / 2
    crop_h = (h - 720) / 2
    cropped = img[math.ceil(crop_h):h - math.floor(crop_h), math.ceil(crop_w):w - math.floor(crop_w)]
    return cropped
