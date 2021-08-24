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
from scipy.spatial import distance
import cv2
from plantcv import plantcv as pcv

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

class ManualDrawPoly:
    def __init__(self, img, lb_im, savename=None):
        self.img = img
        self.lb_im = lb_im
        self.savename = savename
        self.points  = []
        self.events  = []

        # create a window
        self.window = tk.Tk()
        self.window.title("Manual Polygon Draw")
        self.window.bind("<Return>", self.on_hit_enter)
        self.window.bind("<KP_Enter>", self.on_hit_enter)
        self.window.bind("<a>", self.on_hit_a)
        self.window.bind("<c>", self.on_hit_c)
        self.window.bind("<s>", self.on_hit_s)
        self.window.bind("<Escape>", self.on_hit_esc)

        # self.window.columnconfigure(0, minsize=800)
        # self.window.columnconfigure(1, minsize=450)
        self.window.columnconfigure(0, minsize=1000)
        self.window.rowconfigure(0, minsize=700)
        self.window.rowconfigure(1, minsize=100)

        # frame for plot
        self.fig = plt.figure(figsize=(8, 4))
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        # self.canvas.mpl_connect("button_press_event", self.onclick)
        fr_plot = self.canvas.get_tk_widget()

        self.ax1 = self.fig.add_subplot(1, 2, 1)
        self.ax2 = self.fig.add_subplot(1, 2, 2)

        # frame for buttons
        fr_buttons = tk.Frame(self.window)

        fr_draw = tk.Frame(fr_buttons)
        self.txt_btn_start = tk.StringVar()
        self.txt_btn_start.set("Show images")
        self.txt_lbl_start = tk.StringVar()
        self.btn_start = tk.Button(fr_draw, textvariable=self.txt_btn_start, command=self.draw_poly)
        self.lbl_start = tk.Label(fr_draw, textvariable=self.txt_lbl_start)
        self.txt_lbl_start.set(' or hit on "c"')
        self.btn_start.grid(row=0, column=0, sticky="ew")
        self.lbl_start.grid(row=0, column=1, sticky="ew")

        lbl_inst = tk.Label(fr_buttons,
                            text='Important!! Please zoom in first if needed, then click on "Start"')

        fr_start = tk.Frame(fr_buttons)
        btn_start = tk.Button(fr_start, text="Start", command=self.start)
        lbl_start = tk.Label(fr_start, text=' (or hit on "a")')
        btn_start.grid(row=0, column=0, sticky="ew")
        lbl_start.grid(row=0, column=1, sticky="ew")

        fr_show = tk.Frame(fr_buttons)
        btn_show= tk.Button(fr_show, text="Finish Drawing", command=self.finish_draw)
        lbl_show = tk.Label(fr_show, text=' (or hit on "Enter")')
        btn_show.grid(row=0, column=0, sticky="ew")
        lbl_show.grid(row=0, column=1, sticky="ew")

        if self.savename is not None:
            fr_save = tk.Frame(fr_buttons)
            fr_save_l = tk.Frame(fr_save)
            btn_save= tk.Button(fr_save_l, text="Save", command=self.save_results)
            lbl_save = tk.Label(fr_save_l, text=' (or hit on "s")')
            btn_save.grid(row=0, column=0, sticky="ew")
            lbl_save.grid(row=0, column=1, sticky="ew")

            self.txt_lbl_info = tk.StringVar()
            lbl_info = tk.Label(fr_save, textvariable=self.txt_lbl_info )
            fr_save_l.grid(row=0, column=0, sticky="ew")
            lbl_info.grid(row=0, column=1, sticky="ew")

        fr_quit= tk.Frame(fr_buttons)
        btn_quit= tk.Button(fr_quit, text="Exit", command=self.quit)
        lbl_quit = tk.Label(fr_quit, text=' (or hit on "Esc")')
        btn_quit.grid(row=0, column=0, sticky="ew")
        lbl_quit.grid(row=0, column=1, sticky="ew")

        fr_draw.grid(row=0, column=0, sticky="ew")
        lbl_inst.grid(row=1, column=0, sticky="ew")
        fr_start.grid(row=2, column=0, sticky="ew")
        fr_show.grid(row=3, column=0, sticky="ew")
        if self.savename is not None:
            fr_save.grid(row=4, column=0, sticky="ew")
        fr_quit.grid(row=5, column=0, sticky="ew")

        fr_plot.grid(row=0, column=0, sticky="nsew")
        fr_buttons.grid(row=1, column=0, sticky="nsew")

        NavigationToolbar2Tk(self.canvas, fr_plot)
        self.window.mainloop()

    def onclick(self, event):
        self.events.append(event)
        if event.button == 1:
            self.ax1.plot(event.xdata, event.ydata, 'x', c='red')
            if len(self.points) > 0:
                p1x, p1y = self.points[-1][0], self.points[-1][1]
                self.ax1.plot([p1x, event.xdata], [p1y, event.ydata], c="red")
            self.points.append((event.xdata, event.ydata))
        else:
            # import pdb
            # pdb.set_trace()
            idx_remove, _ = _find_closest((event.xdata, event.ydata), self.points)
            # print(idx_remove)
            # remove the closest point to the user's right-clicked one
            self.points.pop(idx_remove)
            ax1plots = self.ax1.lines
            self.ax1.lines.remove(ax1plots[idx_remove*2])
            self.ax1.lines.remove(ax1plots[idx_remove*2-1])
        self.fig.canvas.draw()
        # print(len(self.points))

    def start(self):
        self.canvas.mpl_connect("button_press_event", self.onclick)

    def draw_poly(self):
        # remove all lines if any
        self.ax1.lines = []

        mask_all = copy.deepcopy(self.lb_im)
        mask_all[np.where(mask_all > 0)] = 1
        ma_lb_im = np.ma.array(self.lb_im, mask=~mask_all.astype(bool))
        self.ax1.imshow(self.img)
        self.ax1.imshow(ma_lb_im)
        self.fig.canvas.draw()

    def finish_draw(self):
        roi_contour = [np.array(self.points, dtype=np.int32)]
        print(np.unique(self.lb_im))
        max_lb = max(np.unique(self.lb_im))
        print(f"\nmaximum index {max_lb}")
        # update label image
        self.lb_im = cv2.drawContours(self.lb_im, roi_contour, 0, max_lb+1, -1)
        print(f"\nupdated maximum index {max(np.unique(self.lb_im))}")

        # mask_all = copy.deepcopy(self.lb_im)
        # mask_all[np.where(mask_all > 0)] = 1
        # ma_lb_im = np.ma.array(self.lb_im, mask=~mask_all.astype(bool))
        # self.ax2.imshow(self.img)
        # self.ax2.imshow(ma_lb_im)
        self.ax2.imshow(self.lb_im)
        self.fig.canvas.draw()

        self.points  = []
        self.events  = []

        if not self.txt_btn_start.get() == "Continue":
            self.txt_btn_start.set("Continue")
            self.txt_lbl_start.set(' or hit on "c" if there are other missing segments')

    def save_results(self):
        pkl.dump(self.lb_im, open(self.savename, "wb"))
        self.txt_lbl_info.set(f"Result saved: \n{self.savename}")

    # keyboard activities
    def on_hit_a(self, event):
        self.start()

    def on_hit_c(self, event):
        self.draw_poly()

    def on_hit_enter(self, event):
        self.finish_draw()

    def on_hit_s(self, event):
        self.save_results()

    def on_hit_esc(self, event):
        self.quit()

    def quit(self):
        self.window.quit()
        self.window.destroy()

if __name__ == "__main__":
    path_lb_im = "/Users/hudanyunsheng/Desktop/ts_labeling/data/plant5/curated_seg_labels"
    path_img = path_lb_im.replace("curated_seg_labels", "images")
    file_lb_im = glob.glob(os.path.join(path_lb_im, "*2019-11-04-19-05_.pkl"))[0]
    file_img = file_lb_im.replace(path_lb_im, path_img).replace(".pkl", ".png")

    lb_im = pkl.load(open(file_lb_im, "rb"))
    print(lb_im.shape)

    img, _, _ = pcv.readimage(file_img)
    print(img.shape)
    manual_polygon_draw = ManualDrawPoly(img, lb_im, savename="/Users/hudanyunsheng/Desktop/ts_labeling/data/plant5/curated_seg_labels/temp/temp.pkl")



