import cv2 as cv
import numpy as np
import os

def preprocess(labels_path, sep_labels_path):
    # list all files on labels_path
    labels_filenames = os.listdir(labels_path)

    count = 0
    for label_filename in labels_filenames:
        label_path = os.path.join(labels_path, label_filename)

        print(f'segmenting {label_filename}')
        masks = segment_labels(label_path)

        for att in masks:
            mask = masks[att]
            path = f"{sep_labels_path}/{label_filename[:-4]}_{att}.png"
            print(f'{count} - writing {path}')
            cv.imwrite(path, mask)

        count += 1
            # cv.imwrite(f'{label_filename[:-4]}_{mask}', mask)


def segment_labels(label_path):
    atts = {
        "background": (0, 0, 0),
        "mouth": (255, 0, 0),
        "eyes": (0, 255, 0),
        "nose": (0, 0, 255),
        "face": (128, 128, 128),
        "hair": (255, 255, 0),
        "eyebrows": (255, 0, 255),
        "ears": (0, 255, 255),
        "teeth": (255, 255, 255),
        "beard": (255, 192, 192),
        "sunglasses": (0, 128, 128),
    }

    label = cv.imread(label_path)
    mask = np.zeros(label.shape, dtype=np.uint8)

    masks = {}

    for att in atts:
        color = atts[att]

        mask = cv.inRange(label, color, color)
        masks[att] = mask
        # cv.imshow(att, mask)
        # cv.waitKey(0)

        # cv.imwrite(f"{sep_labels_path}/{label_path[:-4]}_{att}.png", mask)

    return masks


# separate_masks("./labels.png")
preprocess("./organized_dataset/labels", "./organized_dataset/segmented_labels")
