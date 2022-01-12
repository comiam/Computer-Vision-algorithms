from tkinter import Tk
from tkinter.filedialog import askopenfilename

import cv2


class Rectangle:
    def __init__(self, x=0, y=0, w=0, h=0):
        self.x = x
        self.y = y
        self.width = w
        self.height = h

    def merge(self, another_rect):
        x = min(self.x, another_rect.x)
        y = min(self.y, another_rect.y)
        width = self.width if self.x == another_rect.x else self.width + another_rect.width
        height = self.height if self.y == another_rect.y else self.height + another_rect.height

        return Rectangle(x, y, width, height)


class Region:
    def __init__(self):
        self.childs = []
        self.hidden = False
        self.region = Rectangle()
        self.std = 0


def merge_regs(img, reg0, reg1, thr=15):
    if len(reg0.childs) != 0 or len(reg1.childs) != 0:
        return False

    rect = reg0.region.merge(reg1.region)
    img_reg = img[rect.y:rect.y + rect.height, rect.x:rect.x + rect.width]

    if judge_std(img_reg, thr):
        reg0.region = rect
        reg0.std = calc_std(img_reg)
        reg1.hidden = True
        return True
    else:
        return False


def merge(img, reg, thr=15):
    if len(reg.childs) == 0:
        return

    for child in reg.childs:
        merge(img, child)

    row1 = merge_regs(img, reg.childs[0], reg.childs[1], thr)
    row2 = merge_regs(img, reg.childs[2], reg.childs[3], thr)

    if not (row1 or row2):
        merge_regs(img, reg.childs[0], reg.childs[2], thr)
        merge_regs(img, reg.childs[1], reg.childs[3], thr)


def split(img, rect, thr=15):
    reg = Region()

    reg.region = rect

    if judge_std(img, thr):
        reg.std = calc_std(img)
    else:
        h = int(img.shape[0] / 2)
        w = int(img.shape[1] / 2)

        reg.childs.append(split(img[0:h, 0:w], Rectangle(x=rect.x, y=rect.y, w=w, h=h)))
        reg.childs.append(split(img[0:h, w:2 * w], Rectangle(x=rect.x + w, y=rect.y, w=w, h=h)))
        reg.childs.append(split(img[h:2 * h, 0:w], Rectangle(x=rect.x, y=rect.y + h, w=w, h=h)))
        reg.childs.append(split(img[h:2 * h, w:2 * w], Rectangle(x=rect.x + w, y=rect.y + h, w=w, h=h)))

    return reg


def calc_std(img):
    mean, std = cv2.meanStdDev(img)
    return std


def judge_std(img, thr):
    std = calc_std(img)

    return std <= thr or img.shape[0] * img.shape[1] <= 25


def draw_region(img, reg: Region):
    if not reg.hidden and len(reg.childs) == 0:
        cv2.rectangle(img, (reg.region.x, reg.region.y),
                      (reg.region.x + reg.region.width, reg.region.y + reg.region.height),
                      (int(255 - 20 * reg.std), 0, int(20 * reg.std)), -1)
    else:
        for child in reg.childs:
            draw_region(img, child)


if __name__ == '__main__':
    Tk().withdraw()
    filename = askopenfilename()

    img = cv2.imread(filename, cv2.IMREAD_COLOR)

    global_area = Rectangle(0, 0, img.shape[1], img.shape[0])
    img_area = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

    reg = split(img_area, global_area)
    merge(img_area, reg)
    img_original = img.copy()
    draw_region(img, reg)

    cv2.imshow('original', img_original)
    cv2.imshow('sm', img)

    cv2.waitKey(0)
