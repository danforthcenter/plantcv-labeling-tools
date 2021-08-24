import matplotlib
matplotlib.use('TkAgg')
import os
import glob
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import random
import pickle as pkl
import numpy as np
from scipy.spatial import distance
import re
import copy
import time
from plantcv import plantcv as pcv


def _find_closest(pt, pts):
    """ Given coordinates of a point and a list of coordinates of a bunch of points, find the point that has the smallest Euclidean to the given point

    :param pt: (tuple) coordinates of a point
    :param pts: (a list of tuples) coordinates of a list of points
    :return: index of the closest point and the coordinates of that point
    """
    if pt in pts:
        return pts.index(pt), pt
    dists = distance.cdist([pt], pts, 'euclidean')
    idx = np.argmin(dists)
    return idx, pts[idx]

class ManualLabeling:
    def __init__(self, path_img, path_gt, pattern_datetime, path_lb_im_cure, list_img=None, ext=".png"):
        self.path_img = path_img
        self.path_gt = path_gt
        if not os.path.exists(path_gt):
            os.makedirs(path_gt)
        self.dt_pattern = pattern_datetime
        self.ext = ext
        if list_img is not None:
            self.list_img = list_img
        else:
            self.list_img = [f for f in os.listdir(path_img) if f.endswith(self.ext)]
        self.tps = []
        for im_name in self.list_img:
            self.tps.append(re.search(pattern_datetime, im_name).group())

        self.tps.sort() # sorted list of timepoints

        self.path_lb_im_cure = path_lb_im_cure #self.path_img.replace("images", "curated_seg_labels")

        self.T = len(self.list_img)
        self.lb_uid = [[] for _ in range(self.T)] # every list represents a lid(local-id)-to-uid(unique-id) correspondence

        # initialize all available unique uids
        self.all_uids = None
        self.t, self.tp_t, self.f_im_t, self.lb_im_t, self.lb_im_t1, self.lb_im_t2 = None, None, None, None, None, None
        self.uids_t, self.cids_t, self.cid_t, self.uids_lbd_t = None, None, None, None

        # create a window
        self.window = None
        self.txt_cid, self.txt_input, self.ent_input, self.txt_uid, self.txt_cids, self.txt_uids_lbd, self.txt_max_uid, self.txt_sv = \
            None, None, None, None, None, None, None, None
        self.canvas1, self.canvas2, self.fig1, self.fig2, self.ax1, self.ax2, self.ax3 = None, None, None, None, None, None, None

        # self.points = None

    def lbl(self, tp_t):
        self.tp_t = tp_t
        self.t = self.tps.index(tp_t)
        self.uids_lbd_t = []
        print(self.path_lb_im_cure)
        self.f_im_t = glob.glob(os.path.join(self.path_lb_im_cure, f"*{tp_t}*"))[0]
        self.lb_im_t = pkl.load(open(self.f_im_t, "rb"))
        if self.t == 0:
            # initialize lb_uid for the 1st tp
            # tp0 = self.tps[0]
            # file_seg = glob.glob(os.path.join(self.path_seg_cure, f"*{tp0}*"))[0]
            # seg = pkl.load(open(file_seg, "rb"))
            num1 = len(np.unique(self.lb_im_t)) - 1
            # num1 = seg["masks"].shape[2]
            self.lb_uid[0] = [i for i in range(num1)]
            self.all_uids = copy.deepcopy(self.lb_uid[0])
            _ = self.save_gt()

        else:
        # if self.t > 0:
            print(f"\n{self.t}")
            print(f"\ninitial: {self.lb_uid}")
            if not self.lb_uid[self.t - 1]:
                self.import_saved()
                print(f"\nloaded {self.lb_uid}")

            f_img_t = glob.glob(os.path.join(self.path_img, f"*{tp_t}*"))[0]
            img_t, _, _ = pcv.readimage(f_img_t)

            self.uids_t = copy.deepcopy(self.all_uids)

            mask_t = copy.deepcopy(self.lb_im_t)
            mask_t [np.where(mask_t  > 0)] = 1
            ma_lb_im_t = np.ma.array(self.lb_im_t, mask=~mask_t.astype(bool))
            num_seg_t = len(np.unique(self.lb_im_t)) - 1
            self.cids_t = [i+1 for i in range(num_seg_t)]
            self.lb_uid[self.t] = [None] * num_seg_t

            # 1 tp before t
            tp_t1 = self.tps[self.t-1]
            f_im_t1 = glob.glob(os.path.join(self.path_lb_im_cure, f"*{tp_t1}*"))[0]
            self.lb_im_t1 = pkl.load(open(f_im_t1, "rb"))
            mask_t1 = copy.deepcopy(self.lb_im_t1)
            mask_t1[np.where(mask_t1 > 0)] = 1
            ma_lb_im_t1 = np.ma.array(self.lb_im_t1, mask=~mask_t1.astype(bool))
            num_seg_t1 = len(np.unique(self.lb_im_t1)) - 1

            f_img_t1 = glob.glob(os.path.join(self.path_img, f"*{tp_t1}*"))[0]
            img_t1, _, _ = pcv.readimage(f_img_t1)

            # create a window
            self.window = tk.Tk()
            self.window.title("Manual Labeling")
            self.window.bind("<Return>", self.on_hit_enter)
            self.window.bind("<KP_Enter>", self.on_hit_enter)
            self.window.bind("<s>", self.on_hit_s)
            self.window.bind("<n>", self.on_hit_n)
            self.window.bind("<r>", self.on_hit_r)

            self.window.columnconfigure(0, minsize=800)
            self.window.columnconfigure(1, minsize=450)
            self.window.rowconfigure(0, minsize=500)
            self.window.rowconfigure(1, minsize=100)

            fr_left = tk.Frame(self.window) # frame on the left, display info for t-1 and t-2
            fr_right = tk.Frame(self.window) # frame on the left, display info for t

            # frames for plots
            self.fig1 = plt.figure(figsize=(8,4))
            self.canvas1 = FigureCanvasTkAgg(self.fig1, master=fr_left)
            self.canvas1.mpl_connect("button_press_event", self.onclick_lbl)

            self.fig2 = plt.figure(figsize=(4,4))
            self.canvas2 = FigureCanvasTkAgg(self.fig2, master=fr_right)
            self.canvas2.mpl_connect("button_press_event", self.onclick_lbl)

            fr_plot1 = self.canvas1.get_tk_widget()  # .pack(side=tk.TOP, fill=tk.BOTH, expand=1)
            fr_plot2 = self.canvas2.get_tk_widget()  #

            # frame for labels of the left frame
            self.txt_uid = tk.StringVar()
            lbl_uid = tk.Label(fr_left, textvariable=self.txt_uid)
            self.txt_uid.set(f"Available unique indices: {self.uids_t}")

            txt_lbl_inst1 = tk.StringVar()
            lbl_inst1 = tk.Label(fr_left, textvariable=txt_lbl_inst1)

            self.txt_cid = tk.StringVar()
            lbl_cid = tk.Label(fr_left, textvariable=self.txt_cid)

            # self.txt_sv = tk.StringVar()
            # lbl_sv = tk.Label(fr_left, textvariable=self.txt_sv)
            # self.txt_sv.set("Click on the 'Save' button or hit on 's' to save results!")

            # assembly the left frame
            fr_plot1.grid(row=0, column=0, sticky="nw")
            lbl_uid.grid(row=1, column=0, sticky="nw")
            lbl_inst1.grid(row=2, column=0, sticky="nw")
            lbl_cid.grid(row=3, column=0, sticky="nw")
            # lbl_sv.grid(row=4, column=0, sticky="nw")

            # frame for labels of the right frame
            self.txt_cids = tk.StringVar()
            lbl_cids = tk.Label(fr_right, textvariable=self.txt_cids)
            self.txt_cids.set(f"Unlabeled leaf indices: {self.cids_t}")

            lbl_inst2 = tk.Label(fr_right, text="Click on the same leaf to assign the same unique id. "
                                                "\nHit on 'Enter' to confirm.")

            self.txt_uids_lbd = tk.StringVar()
            lbl_uids_lbd = tk.Label(fr_right, textvariable=self.txt_uids_lbd)

            self.txt_max_uid = tk.StringVar()
            lbl_max_uid = tk.Label(fr_right, textvariable=self.txt_max_uid)

            # (sub-)frame for input (sub-frame for fr_labels)
            fr_input = tk.Frame(fr_right)
            self.txt_input = tk.StringVar()
            lbl_input = tk.Label(fr_input, textvariable=self.txt_input)
            self.ent_input = tk.Entry(master=fr_input, width=10)

            # assembly the input (sub-)frame
            lbl_input.grid(row=0, column=0, sticky="e")
            self.ent_input.grid(row=0, column=1)

            fr_restart = tk.Frame(fr_right)
            btn_restart = tk.Button(fr_restart, text="Start Over", command=self.start_over)
            lbl_restart = tk.Label(fr_restart, text='(or hit on "r")')
            btn_restart.grid(row=0, column=0, sticky="ew")
            lbl_restart.grid(row=0, column=1, sticky="ew")

            # assembly the right frame
            fr_plot2.grid(row=0, column=0, sticky="nw")
            lbl_cids.grid(row=1, column=0, sticky="nw")
            lbl_inst2.grid(row=2, column=0, sticky="nw")
            fr_input.grid(row=3, column=0, sticky="nw")
            lbl_uids_lbd.grid(row=4, column=0, sticky="nw")
            lbl_max_uid.grid(row=5, column=0, sticky="nw")
            fr_restart.grid(row=6, column=0, sticky="nw")

            # frame for buttons
            fr_buttons = tk.Frame(self.window)
            # btn_finish = tk.Button(fr_buttons, text="Save", command=self.finish_lbl)
            fr_save = tk.Frame(fr_buttons)
            btn_save = tk.Button(fr_save, text="Save", command=self.finish_lbl)
            lbl_save = tk.Label(fr_save, text='(or hit on "s")')
            btn_save.grid(row=0, column=0, sticky="ew")
            lbl_save.grid(row=0, column=1, sticky="ew")

            self.txt_sv = tk.StringVar()
            lbl_save_info = tk.Label(fr_save, textvariable=self.txt_sv)

            fr_exit = tk.Frame(fr_buttons)
            btn_exit = tk.Button(fr_exit, text="Next", command=self.quit)
            lbl_exit = tk.Label(fr_exit, text='(or hit on "n")')
            btn_exit.grid(row=0, column=0, sticky="ew")
            lbl_exit.grid(row=0, column=1, sticky="ew")

            # btn_exit = tk.Button(fr_buttons, text="Exit", command=self.quit)
            # assembly the button frame
            fr_save.grid(row=1, column=0, sticky="ew")  # , padx=5, pady=5)
            lbl_save_info.grid(row=2, column=2, sticky="ew")  # , padx=5, pady=5)
            fr_exit.grid(row=3, column=0, sticky="ew")  # , padx=5)
            # fr_restart.grid(row=4, column=0, sticky="ew")  # , padx=5)

            # assembly all frames
            fr_left.grid(row=0, column=0, sticky="nsew")
            fr_right.grid(row=0, column=1, sticky="nsew")
            fr_buttons.grid(row=1, column=0, sticky="nsew")

            if self.t > 1:
                # 2 tps before t
                tp_t2 = self.tps[self.t - 2]
                f_im_t2 = glob.glob(os.path.join(self.path_lb_im_cure, f"*{tp_t2}*"))[0]
                self.lb_im_t2 = pkl.load(open(f_im_t2, "rb"))
                num_seg_t2 = len(np.unique(self.lb_im_t2)) - 1
                mask_t2 = copy.deepcopy(self.lb_im_t2)
                mask_t2[np.where(mask_t2 > 0)] = 1
                ma_lb_im_t2 = np.ma.array(self.lb_im_t2, mask=~mask_t2.astype(bool))

                f_img_t2 = glob.glob(os.path.join(self.path_img, f"*{tp_t2}*"))[0]
                img_t2, _, _ = pcv.readimage(f_img_t2)

                self.ax1 = self.fig1.add_subplot(1, 2, 1)
                # self.ax1.imshow(self.lb_im_t2)
                self.ax1.imshow(img_t2)
                self.ax1.imshow(ma_lb_im_t2)
                self.ax1.set_title(tp_t2)
                txt_lbl_inst1.set(f"Click on a leaf to check unique leaf index information for {tp_t1} or {tp_t2}.")
            else:
            # elif self.t == 1:
                txt_lbl_inst1.set(f"Click on a leaf to check unique leaf index information for {tp_t1}.")

            self.ax2 = self.fig1.add_subplot(1, 2, 2)
            # self.ax2.imshow(self.lb_im_t1)
            self.ax2.imshow(img_t1)
            self.ax2.imshow(ma_lb_im_t1)
            self.ax2.set_title(tp_t1)

            self.ax3 = self.fig2.add_subplot(1, 1, 1)
            # self.ax3.imshow(self.lb_im_t)
            self.ax3.imshow(img_t)
            self.ax3.imshow(ma_lb_im_t)
            self.ax3.set_title(tp_t)

            self.ax3.set_title(tp_t)
            self.window.mainloop()
        plt.close("all")


    # keyboard activities
    def onclick_lbl(self, event):
        if event.button == 1:
            x, y = event.xdata, event.ydata
            ax = event.inaxes
            ax.plot(x, y, "x", c="red")
            # global idx_t1, idx_t
            if ax in [self.ax1]:
                t2 = self.t-2
                cid_t2 = int(self.lb_im_t2[int(y), int(x)])

                if cid_t2 == 0:
                    self.txt_cid.set("Clicked on a non-leaf area, please try again.")
                else:
                    uid_t2 = int(self.lb_uid[t2][cid_t2-1])
                    # self.txt_cid.set(f"In {self.tps[t2]}, the unique id for clicked leaf {cid_t2} is {uid_t2}.")
                    self.txt_cid.set(f"In {self.tps[t2]}, the unique id for selected leaf is {uid_t2}.")
                    self.ent_input.delete(0, tk.END)
                    self.ent_input.insert(0, f'{uid_t2}')

            elif ax in [self.ax2]:
                t1 = self.t-1
                # print(t1)
                cid_t1 = int(self.lb_im_t1[int(y), int(x)])
                if cid_t1 == 0:
                    self.txt_cid.set("Clicked on a non-leaf area, please try again.")
                else:
                    uid_t1 = int(self.lb_uid[t1][cid_t1-1])
                    # self.txt_cid.set(f"In {self.tps[t1]}, the unique id for clicked leaf {cid_t1} is {uid_t1}.")
                    self.txt_cid.set(f"In {self.tps[t1]}, the unique id for selected leaf is {uid_t1}.")
                    self.ent_input.delete(0, tk.END)
                    self.ent_input.insert(0, f'{uid_t1}')
            elif ax in [self.ax3]:
                self.cid_t = int(self.lb_im_t[int(y), int(x)])
                if self.cid_t == 0:
                    self.txt_input.set("Clicked on a non-leaf area, please try again.")
                else:
                    self.txt_input.set(f"Assign unique id for clicked leaf {self.cid_t}: ")
            else:
                pass
        self.canvas1.draw()
        self.canvas2.draw()

    def on_hit_enter(self, event):
        uid = int(self.ent_input.get())
        if uid not in self.uids_lbd_t:
            self.uids_lbd_t.append(uid)
        self.ent_input.delete(0, tk.END)

        self.txt_uids_lbd.set(f"Labeled unique indices in {self.tps[self.t]}: {self.uids_lbd_t}")

        if uid in self.uids_t:
            self.uids_t.remove(uid)
        # if self.cid_t not in self.cids_t:
        if self.cid_t in self.cids_t:
            self.cids_t.remove(self.cid_t)

        # update displayed labels
        self.txt_uid.set(f"Available unique indices: {self.uids_t}")
        self.txt_cids.set(f"Unlabeled leaf indices: {self.cids_t}")

        # append to lb_uid
        # print(f"\nt: {self.t}")
        # print(f"\nlb_uid (t-1): {self.lb_uid[self.t]}")
        self.lb_uid[self.t][self.cid_t-1] = uid

        # print(f"\nlb_uid (t-1): {self.lb_uid[self.t-1]}")
        # print(f"\nlb_uid (t): {self.lb_uid[self.t]}")

        # append to all_uids is not already in
        if uid not in self.all_uids:
            self.all_uids.append(uid)
        self.txt_max_uid.set(f"Maximum unique indices in whole time series til now: {max(self.all_uids)}.")

    def on_hit_s(self, event):
        self.finish_lbl()

    def on_hit_n(self, event):
        self.quit()

    def on_hit_r(self, event):
        self.start_over()


    def finish_lbl(self):
        file_gt = self.save_gt()
        # display a message saying the result has been saved
        if os.path.isfile(file_gt):
            self.txt_sv.set(f"Result saved!"
                            f"\n({file_gt})")
        # time.sleep(5)
        # # automatically quit after 5 seconds
        # self.quit()

    def start_over(self):
        self.quit()
        self.lbl(self.tp_t)

    def save_gt(self):
        file_gt = self.f_im_t.replace(self.ext, ".pkl").replace(self.path_lb_im_cure, self.path_gt)
        to_save = {"lb_uid": self.lb_uid,
                   "all_uids": self.all_uids}
        # print(f"Save at {os.path.join(self.path_gt, file_gt)}")
        # print(f"Save at {self.path_gt}, {file_gt}")
        pkl.dump(to_save, open(file_gt, "wb"))
        return file_gt

    def import_saved(self):
        tp_t1 = self.tps[self.t - 1]
        f_im_t1 = glob.glob(os.path.join(self.path_lb_im_cure, f"*{tp_t1}*"))[0]
        file_gt_t1 = f_im_t1.replace(self.ext, ".pkl").replace(self.path_lb_im_cure, self.path_gt)
        if os.path.isfile(file_gt_t1):
            print(f"\nload result from {file_gt_t1}")
            loaded = pkl.load(open(file_gt_t1, "rb"))
            for key, value in loaded.items():
                setattr(self, key, value)
        else:
            print("\nNot available!")


    def quit(self):
        self.window.quit()
        self.window.destroy()

if __name__ == "__main__":
    pidx = 0
    dir_img = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/images"
    dir_cure_lb_im = f"/Users/hudanyunsheng/Documents/github/plantcv-labeling-tools/time_series_labeling/sample/data/plant{pidx}/curated_seg_labels"
    pattern_dt = "\d{4}-\d{2}-\d{2}-\d{2}-\d{2}"
    ext = ".png"
    dir_gt = dir_img.replace("data", "ground_truth").replace("images", "")
    # dir_gt = os.path.join(dir_gt, "today")
    list_img_ = [f for f in os.listdir(dir_img) if f.endswith(ext)]
    list_img_.sort()
    list_img = list_img_
    manual_labeling = ManualLabeling(dir_img, dir_gt, pattern_dt, dir_cure_lb_im, list_img=list_img, ext=ext)

    tps = []
    for img_name in list_img:
        tps.append(re.search(pattern_dt, img_name).group())
    tps.sort()  # sorted list of timepoints
    # print(len(tps))

    # print(f"\nInitial: {manual_labeling.lb_uid}")
    ind = 0
    # if starting from the middle you have to make sure you have the label available for t-1 saved in dir_gt
    # the example below starts from 2019-11-03-09-05
    # if start from the 1st time point, comment the code below
    ind = tps.index("2019-10-26-12-05")
    for tp in tps[ind:]:
        print(f"\nSelect timepoint: {tp}")
        manual_labeling.lbl(tp)

        print(f"\nUpdated {manual_labeling.lb_uid}")


