import matplotlib
matplotlib.use('TkAgg')
import os
import glob
import cv2
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
# import matplotlib.backends.backend_tkagg as tkagg
import random
import pickle as pkl
import numpy as np
from scipy.spatial import distance
import re
import copy
from manual_draw_polygon import ManualDrawPoly


def _find_closest(pt, pts):
    """ Given coordinates of a point, and a list of coordinates of a bunch of points,
    find the point that has the smallest Euclidean to the given point

    :param pt: (tuple) coordinates of a point
    :param pts: (a list of tuples) coordinates of a list of points
    :return: index of the closest point and the coordinates of that point
    """
    if pt in pts:
        return pts.index(pt), pt
    dists = distance.cdist([pt], pts, 'euclidean')
    idx = np.argmin(dists)
    return idx, pts[idx]

class ManualRemRedun:
    def __init__(self, path_img, pattern_datetime, path_seg, path_lb_im, list_img, path_seg_cure=None, path_lb_im_cure=None):
        self.path_img = path_img
        self.dt_pattern = pattern_datetime
        self.list_img = list_img#[f for f in os.listdir(path_img) if f.endswith(ext)]
        self.tps = []
        for im_name in self.list_img:
            self.tps.append(re.search(pattern_datetime, im_name).group())

        self.tps.sort() # sorted list of timepoints

        self.path_seg = path_seg
        self.path_lb_im = path_lb_im
        self.path_lb_im_vis = os.path.join(self.path_lb_im, "visualization")
        if path_seg_cure is not None:
            self.path_seg_cure = path_seg_cure
        else:
            self.path_seg_cure = os.path.join(self.path_seg, "curated_segmentation")
        if path_lb_im_cure is not None:
            self.path_lb_im_cure = path_lb_im_cure
        else:
            self.path_lb_im_cure = os.path.join(self.path_lb_im, "curated_seg_labels")
        self.path_lb_im_cure_vis = os.path.join(self.path_lb_im_cure, "visualization")
        # self.path_lb_im = self.path_img.replace("images", "seg_labels")
        # self.path_seg_cure = self.path_img.replace("images", "curated_segmentation")
        # self.path_lb_im_cure = self.path_img.replace("images", "curated_seg_labels")

        if not os.path.exists(self.path_lb_im):
            os.makedirs(self.path_lb_im)
        if not os.path.exists(self.path_seg_cure):
            os.makedirs(self.path_seg_cure)
        if not os.path.exists(self.path_lb_im_cure):
            os.makedirs(self.path_lb_im_cure)
        if not os.path.exists(self.path_lb_im_cure_vis):
            os.makedirs(self.path_lb_im_cure_vis)
        if not os.path.exists(self.path_lb_im_vis):
            os.makedirs(self.path_lb_im_vis)

        self.tp, self.img = None, None
        self.seg, self.lb_im, self.seg_cure, self.lb_im_cure  = None, None, None, None
        self.remove_ind = None
        self.points = None
        self.fig, self.ax1, self.ax2, self.ax3 = None, None, None, None
        self.canvas = None
        self.txt_lbl_note, self.txt_lbl_mark = None, None

        self.file_missing = os.path.join(self.path_lb_im_cure, "missing.csv")
        # create a window
        self.window = None



    @staticmethod
    def mask_to_lbl_im(masks):
        r, c, n = masks.shape
        label_im = np.zeros((r, c))
        for i in range(n):
            label_im[np.where(masks[:,:,i]==1)] = i+1
        return label_im

    def rem_redundent(self, tp_t):
        self.tp = tp_t

        self.window = tk.Tk()
        self.window.title("Remove Redundant segments")
        self.window.bind("<Return>", self.on_hit_enter)
        self.window.bind("<KP_Enter>", self.on_hit_enter) # two "Enter" keys
        self.window.bind("<Escape>", self.on_hit_esc)
        self.window.bind("<a>", self.on_hit_a)
        self.window.bind("<s>", self.on_hit_s)
        self.window.columnconfigure(0, minsize=1000)
        self.window.rowconfigure(0, minsize=700)
        self.window.rowconfigure(1, minsize=100)

        self.fig = plt.figure(figsize=(10, 5))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        # self.canvas.mpl_connect("button_press_event", self.onclick_rem)

        # frame for plot
        fr_plot =  self.canvas.get_tk_widget()#.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        # frame for buttons
        fr_buttons = tk.Frame(self.window)

        lbl_inst = tk.Label(fr_buttons, text='Important!! Please zoom in first is needed, then click on "Start"')

        fr_start = tk.Frame(fr_buttons)
        btn_start = tk.Button(fr_start, text="Start", command=self.start)
        lbl_start = tk.Label(fr_start, text=' (or hit on "a")')
        btn_start.grid(row=0, column=0, sticky="ew")
        lbl_start.grid(row=0, column=1, sticky="ew")

        fr_update = tk.Frame(fr_buttons)
        btn_update = tk.Button(fr_update, text="Check updated", command=self.update_rem)
        lbl_update = tk.Label(fr_update, text=' (or hit on "Enter")')
        btn_update.grid(row=0, column=0, sticky="ew")
        lbl_update.grid(row=0, column=1, sticky="ew")

        fr_mark = tk.Frame(fr_buttons)
        btn_mark = tk.Button(fr_mark, text="Click HERE", command=self.draw_poly)
        lbl_mark = tk.Label(fr_mark, text="if there are missing segments")
        btn_mark.grid(row=0, column=0, sticky="ew")
        lbl_mark.grid(row=0, column=1, sticky="ew")

        fr_finish = tk.Frame(fr_buttons)
        btn_finish = tk.Button(fr_finish, text="Save", command=self.finish_rem_add)
        lbl_finish = tk.Label(fr_finish, text=' (or hit on "s")')
        btn_finish.grid(row=0, column=0, sticky="ew")
        lbl_finish.grid(row=0, column=1, sticky="ew")

        fr_exit = tk.Frame(fr_buttons)
        btn_exit = tk.Button(fr_exit, text="Exit", command=self.quit)
        lbl_exit = tk.Label(fr_exit, text=' (or hit on "Esc")')
        btn_exit.grid(row=0, column=0, sticky="ew")
        lbl_exit.grid(row=0, column=1, sticky="ew")

        self.txt_lbl_note = tk.StringVar()
        lbl_note = tk.Label(fr_buttons, textvariable=self.txt_lbl_note)

        self.txt_lbl_mark = tk.StringVar()
        lbl_mark_note = tk.Label(fr_buttons, textvariable=self.txt_lbl_mark)

        lbl_inst.grid(row=0, column=0, sticky="ew")
        fr_start.grid(row=1, column=0, sticky="ew")
        fr_update.grid(row=2, column=0, sticky="ew")  # , padx=5, pady=5)
        fr_mark.grid(row=3, column=0, sticky="ew")  # , padx=5, pady=5)
        fr_finish.grid(row=4, column=0, sticky="ew")  # , padx=5, pady=5)
        fr_exit.grid(row=5, column=0, sticky="ew")#, padx=5)
        lbl_mark_note.grid(row=1, column=1, sticky="ew")
        lbl_note.grid(row=2, column=1, sticky="ew")

        fr_plot.grid(row=0, column=0, sticky="nsew")
        fr_buttons.grid(row=1, column=0, sticky="nsew")

        self.points = []
        self.remove_ind = []

        # convert from segmentation to label image
        print(self.path_seg)
        file_seg = glob.glob(os.path.join(self.path_seg, f"*{tp_t}*"))[0]
        print(file_seg)

        # file_missing = os.path.join(self.path_lb_im_cure, "missing.csv")
        self.seg = pkl.load(open(file_seg, "rb"))
        masks = self.seg["masks"]
        self.lb_im = ManualRemRedun.mask_to_lbl_im(masks)
        file_lbl_im = file_seg.replace(self.path_seg, self.path_lb_im)
        pkl.dump(self.lb_im, open(file_lbl_im, "wb"))

        # initialization for curated segmentation and label image
        self.seg_cure = copy.copy(self.seg)
        self.lb_im_cure = copy.copy(self.lb_im)

        fig, ax = plt.subplots()
        ax.imshow(self.lb_im)
        plt.savefig(file_lbl_im.replace(self.path_lb_im, self.path_lb_im_vis).replace(".pkl", ".png"))
        plt.close("all")

        num_seg = len(np.unique(self.lb_im))-1

        file_img = glob.glob(os.path.join(self.path_img, f"*{tp_t}*"))[0]
        self.img = plt.imread(os.path.join(self.path_img, file_img))
        self.ax1 = self.fig.add_subplot(1, 3, 1)
        self.ax1.imshow(self.img)
        self.ax1.set_title(tp_t)

        self.ax2 = self.fig.add_subplot(1, 3, 2)
        # self.ax2.imshow(self.lb_im)
        self.ax2.imshow(self.img)
        mask_all = copy.deepcopy(self.lb_im)
        mask_all[np.where(mask_all > 0)] = 1
        ma_lb_im = np.ma.array(self.lb_im, mask=~mask_all.astype(bool))
        self.ax2.imshow(ma_lb_im)
        self.ax2.set_title(f"{num_seg} segments\nclick on unwanted segments")

        self.ax3 = self.fig.add_subplot(1, 3, 3)
        self.ax3.set_title("updated label image")
        NavigationToolbar2Tk(self.canvas, fr_plot)
        # tkagg.NavigationToolbar2Tk(self.canvas, fr_plot)
        self.window.mainloop()

    def onclick_rem(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            self.ax2.plot(x, y, "x", c="red")
            self.points.append((event.xdata, event.ydata))
            idx = self.lb_im[int(y), int(x)]
            # print(f"\nclicked on {x} {y}\nindex:{int(idx)}")
            self.remove_ind.append(int(idx-1))
        else:
            print(_find_closest((event.xdata, event.ydata), self.points))
            idx_remove, _ = _find_closest((event.xdata, event.ydata), self.points)
            print(idx_remove)
            self.points.pop(idx_remove)
            self.remove_ind.pop(idx_remove)
            # if len(self.points) > 0:
            #     self.x, self.y = self.points[-1]
            axplots = self.ax2.lines
            self.ax2.lines.remove(axplots[idx_remove])
        self.canvas.draw()

    # buttons
    def start(self):
        self.canvas.mpl_connect("button_press_event", self.onclick_rem)

    def update_rem(self):
        # updated segmentation to be saved
        if len(self.remove_ind) > 0:
            self.remove_ind.sort(reverse=True)
            # masks, rois, class_ids, scores = self.seg["masks"], self.seg["rois"], self.seg["class_ids"], self.seg["scores"]
            # for ind in self.remove_ind:
            self.seg_cure["masks"] = np.delete(self.seg_cure["masks"], self.remove_ind, axis=2)
            self.seg_cure["rois"] = np.delete(self.seg_cure["rois"], self.remove_ind, axis=0)
            self.seg_cure["class_ids"] = np.delete(self.seg_cure["class_ids"], self.remove_ind, axis=0)
            self.seg_cure["scores"] = np.delete(self.seg_cure["scores"], self.remove_ind, axis=0)
            self.lb_im_cure = ManualRemRedun.mask_to_lbl_im(self.seg_cure["masks"])
        self.ax3.imshow(self.lb_im_cure)
        num_seg = len(np.unique(self.lb_im_cure)) - 1
        self.ax3.set_title(f"updated label image\n{num_seg} segments")
        self.canvas.draw()
        self.remove_ind = []

    def draw_poly(self):
        with open(self.file_missing, "a", newline="") as f:
            f.write(f"\n{self.tp}")
        self.txt_lbl_mark.set("Information recorded!")

        img = copy.deepcopy(self.img)
        lb_im = copy.deepcopy(self.lb_im_cure)
        manual_polygon_draw = ManualDrawPoly(img, lb_im)
        self.lb_im_cure = manual_polygon_draw.lb_im

        # update segmentation (if necessary)
        inds = np.unique(self.lb_im_cure)[1:]
        n = len(inds)
        masks, class_ids, scores, rois = self.seg_cure["masks"], self.seg_cure["class_ids"], \
                                         self.seg_cure["scores"], self.seg_cure["rois"]
        r, c, n_ = masks.shape
        if n_ < n:
            # print(f"\n{f_seg_}")
            for ind in inds[n_:]:
                mask_i = np.zeros((r, c), dtype=bool)
                mask_i[np.where(lb_im == ind)] = 1
                # import pdb
                # pdb.set_trace()
                # roi_contour, roi_hierarchy = cv2.findContours(mask_i.astype(np.uint8) * 255, cv2.RETR_TREE,
                #                                               cv2.CHAIN_APPROX_NONE)[-2:]
                # cnt = roi_contour[0]
                # rect = cv2.minAreaRect(cnt)
                #
                # box = cv2.boxPoints(rect)
                # box = np.int0(box)
                masks = np.dstack((masks, mask_i))
                class_ids = np.append(class_ids, 1)
                scores = np.append(scores, np.nan)
                rois = np.vstack((rois, np.ones(rois[0].shape) * np.nan))
            self.seg_cure = {"masks":masks, "rois":rois, "class_ids":class_ids, "scores": scores}

    def finish_rem_add(self):
        if len(self.ax3.images) == 0:
            self.update_rem()
        file_seg = glob.glob(os.path.join(self.path_seg, f"*{self.tp}*"))[0]
        file_lbl_im_cure = file_seg.replace(self.path_seg, self.path_lb_im_cure)
        pkl.dump(self.lb_im_cure, open(file_lbl_im_cure, "wb"))

        fig, ax = plt.subplots()
        ax.imshow(self.lb_im_cure)
        plt.savefig(file_lbl_im_cure.replace(self.path_lb_im_cure, self.path_lb_im_cure_vis).replace(".pkl", ".png"))
        plt.close("all")

        file_seg_cure = file_seg.replace(self.path_seg, self.path_seg_cure)
        pkl.dump(self.seg_cure, open(file_seg_cure, "wb"))

        self.txt_lbl_note.set(f"{file_seg_cure} saved!")

    def quit(self):
        self.window.quit()
        self.window.destroy()

    # key activities
    def on_hit_a(self, event):
        self.start()

    def on_hit_enter(self, event):
        self.update_rem()

    def on_hit_s(self, event):
        self.finish_rem_add()

    def on_hit_esc(self, event):
        self.quit()



if __name__ == "__main__":
    pattern_dt = "\d{4}-\d{2}-\d{2}-\d{2}-\d{2}"

    # directories of data and segmentation
    pidx = 12
    dir_img = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/images"
    dir_sg = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/segmentation"

    # saving directories (if not given, will be set automatically)
    dir_im_lb = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/seg_labels"
    dir_seg_cure = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/curated_segmentation"
    dir_lb_im_cure = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/curated_seg_labels"

    list_end_tp = ["09-05", "12-05", "16-05", "19-05"]
    im_list_ = [f for f in os.listdir(dir_img) if f.endswith(".png")]
    im_list = []
    for img_name in im_list_:
        tp_ = re.search(pattern_dt, img_name).group()
        tp = tp_[-5:]
        if tp in list_end_tp:
            im_list.append(img_name)
    print(f"length of image list: {len(im_list)}")
    manual_labeling = ManualRemRedun(dir_img, pattern_dt, dir_sg, dir_im_lb, im_list, path_seg_cure=dir_seg_cure, path_lb_im_cure=dir_lb_im_cure)
    # manual_labeling = ManualRemRedun(dir_img, pattern_dt, dir_sg, dir_im_lb, im_list)
    tps = []
    for img_name in im_list:
        tps.append(re.search(pattern_dt, img_name).group())

    tps.sort()  # sorted list of timepoints
    print(f"\nTotal number of timepoints: {len(tps)}")
    # print(f"\n{tps}")
    ind = 0
    # if starting from the middle you have to make sure you have the label available for t-1 saved in dir_gt
    # the example below starts from 2019-11-03-09-05
    # if start from the 1st time point, discard the code below
    ind = tps.index("2019-11-01-09-05")
    for tp in tps[ind:]:
    # for tp in [tps[ind]]:
        print(f"\nNow labeling: {tp}")
        manual_labeling.rem_redundent(tp)

