"""
This module is an example of a barebones QWidget plugin for napari

It implements the Widget specification.
see: https://napari.org/stable/plugins/guides.html?#widgets

Replace code below according to your needs.
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget, QLineEdit, QFileDialog, QComboBox, QGridLayout, QLabel
import os
import glob
import numpy as np
import pandas as pd
import subprocess
import tifffile
import xml.etree.ElementTree as ET
from matplotlib.path import Path
import cv2
from PIL import Image
import skimage.data as data
import napari
from dask.array.image import imread

if TYPE_CHECKING:
    import napari


def get_metadata(fname):
    #img = AICSImage(fname)
    img = bfio.BioReader(fname, backend='bioformats')
    sx = img.metadata.images[0].pixels.size_x
    sy = img.metadata.images[0].pixels.size_y
    xdim = img.metadata.images[0].pixels.physical_size_x
    ydim = img.metadata.images[0].pixels.physical_size_y
    px = img.metadata.images[0].pixels.planes[0].position_x
    py = img.metadata.images[0].pixels.planes[0].position_y
    img.close()
    return np.array([fname, sx, sy, xdim, ydim, px, py])

def make_points_along_line(point1, point2, number=10):
    pt_list = []
    delpt = point2-point1
    dist = np.sqrt(np.dot(delpt, delpt).sum())
    delpt = delpt / dist
    spacing = dist/number
    for i in range(0, number+1):
        pt_list.append(point1 + i*spacing*delpt)
    return np.array(pt_list)

def get_values(folder):
    tree = ET.parse(folder + 'Layer-source-metadata.xml')
    tiffs = glob.glob(folder + 'CH*_Z.tif')
    tiffs.sort()
    #img = np.array([tifffile.imread(tiff) for tiff in tiffs])
    img = imread(folder + 'CH*_Z.tif')
    root = tree.getroot()
    for elem in root.iter():
        if ('ImageAxis0Origin' in elem.tag):
            x0 = float((elem.attrib['value']).split(' ')[0])
        if ('ImageAxis1Origin' in elem.tag):
            y0 = float((elem.attrib['value']).split(' ')[0])
        if ('ImageAxis0Resolution' in elem.tag):
            xres = float((elem.attrib['value']).split(' ')[0])
    return img.max(axis=1), x0, y0, xres

def get_all_folders(parent_folder):
    imgs = []
    vals = []
    fnames = glob.glob(parent_folder + '/*Layer*')
    for folder in fnames:
        img, x0, y0, xres = get_values(folder+'/')
        imgs.append(img)
        vals.append([x0, y0, xres])
    vals = np.array(vals)
    x = (vals[:,0] - np.min(vals[:,0]))*1E6/np.median(vals[:,2])
    y = (vals[:,1] - np.min(vals[:,1]))*1E6/np.median(vals[:,2])

    x = x.astype(int)
    y = y.astype(int)

    xx = y
    y = x
    x = xx
    return imgs, x, y

def backsub(inp):
    filterSize =(60, 60)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                    filterSize)
    blurred = cv2.GaussianBlur(inp, (5, 5), 0)
    tophat_img = cv2.morphologyEx(blurred,
                                cv2.MORPH_TOPHAT,
                                kernel)
    rtn = inp.astype(np.single) - (blurred-tophat_img)
    rtn = np.clip(rtn, 0, np.inf)

    return rtn

def clip_image(inp, min, max):
    inp = inp.astype(float)
    inp = inp - min
    inp = inp / (max-min)
    inp = np.clip(inp, 0, 1)
    inp = inp * 255
    #return inp.astype(np.uint8)
    return inp

def get_order_of_images(lines, viewer):
    pts_list = []
    for lin in lines:
        for a in range(0, len(lin)-1):
            point1 = lin[a]
            point2 = lin[a+1]
            points = make_points_along_line(point1, point2, number=30)
            pts_list = pts_list + points.tolist()

    pts_list = np.array(pts_list)
    positions = []
    #for pt in viewer.layers[-1].data:
    for pt in pts_list:
        df_pos = -1
        z = pt[0]
        rois = np.array(viewer.layers[-2].data)
        c_rois = rois[rois[:,:,0]==z].reshape(-1,4,3)[:,:,[1,2]]
        for a, rec in enumerate(c_rois):
            path = Path(rec)
            contained=path.contains_points([pt[1:3]])
            if np.any(contained):
                df_pos = a
        positions.append([z,df_pos])
    positions=np.array(positions)
    positions = positions.astype(int)

    read_list = []
    for p in positions:
        if p[1] != -1:
            read_list.append(p[1])

    read_list = pd.unique(read_list)
    return read_list

class VSISelectQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        load_directory_btn = QPushButton("Choose folder")
        load_directory_btn.clicked.connect(self._on_choose_directory)
        export_vsi_btn = QPushButton("Export VSI")
        export_vsi_btn.clicked.connect(self._on_export_vsi)
        load_overview_btn = QPushButton("Load Overview")
        load_overview_btn.clicked.connect(self._on_load_overview)
        save_highres_btn = QPushButton("Save High Res")
        save_highres_btn.clicked.connect(self._on_save_highres)
        self.text_field = QLineEdit(self)

        self.dropdown_menu = QComboBox(self)
        self.dropdown_menu.addItem("Original")
        self.dropdown_menu.addItem("2X")
        self.dropdown_menu.addItem("4X")
        self.dropdown_menu.addItem("8X")
        self.dropdown_menu.currentIndexChanged.connect(self.dropdown_menu_index_changed)

        self.setLayout(QGridLayout())
        self.layout().addWidget(load_directory_btn, 0, 0)
        self.layout().addWidget(self.text_field, 0, 1)
        self.layout().addWidget(self.dropdown_menu, 1, 0)
        self.layout().addWidget(export_vsi_btn, 1, 1)
        self.layout().addWidget(load_overview_btn, 2, 0)
        self.layout().addWidget(save_highres_btn, 2, 1)
        
        

    def _on_choose_directory(self):
        
        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.root_directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)

        self.text_field.setText(self.root_directory)

    def _on_export_vsi(self):
        # Setup the resolution
        res_dict = {'Original':1, '2X':2, '4X':3, '8X':4}
        res_text = self.dropdown_menu.currentText()
        #resolution = '4X'
        res = res_dict[res_text]

        # Call Imagej macro to export the images
        command = "U:\\Fiji\\Fiji.app\\ImageJ-win64.exe -macro U:\\smc\\Fiji_2016.app\\macros\\ExportVSIForNapari.ijm " + self.root_directory + str(res)
        subprocess.run(command, shell=True)
    
    def dropdown_menu_index_changed(self):
        print(self.dropdown_menu.currentText())
    
    def _on_load_overview(self):
        file_path = self.root_directory + "/*/*.vsi"
        files = glob.glob(file_path)
        files = np.sort(files)
        
        # Get the metadatas from all the files
        metadata = np.array([get_metadata(f) for f in files])

        # Create dataframe with all of the useful metadata
        df = pd.DataFrame(metadata, columns=['fname', 'sx', 'sy', 'xdim', 'ydim', 'px', 'py'])
        df['px'] = df['px'].astype(float)
        df['py'] = df['py'].astype(float)
        df['xdim'] = df['xdim'].astype(float)
        df['ydim'] = df['ydim'].astype(float)
        df['sx'] = df['sx'].astype(int)
        df['sy'] = df['sy'].astype(int)

        # Currently hardcoded overview downsample factor
        overview_downsample = 8

        # Write scaled versions of useful metadata
        df['Mx'] = (df['px']-df['px'].min()) 
        df['My'] = (df['py']-df['py'].min())

        xs = df.iloc[0]['xdim']
        ys = df.iloc[0]['ydim']

        df['Ix'] = df['Mx'] / xs / overview_downsample
        df['Iy'] = df['My'] / ys / overview_downsample

        df['sdx'] = df['sx'] / overview_downsample / (xs/df['xdim'])
        df['sdy'] = df['sy'] / overview_downsample / (ys/df['ydim'])

        df.to_csv(self.root_directory + 'metadata.csv', index=False)
        df['ShortName'] = df['fname'].str.split('\\').str[-2]
        df['Tray'] = df['ShortName'].str[0:2]
        df['Slide'] = df['ShortName'].str[3:5]
        df['Object'] = df['ShortName'].str[6:8]
        df['TraySlide'] = (df['Tray'].astype(int)-1) * 100 + df['Slide'].astype(int)-1
        df['ImgPath'] = df['fname'].str.split('\\').str[0] + '/' + df['ShortName'].str[0:8] + '.tif'
        df.to_csv(self.root_directory + 'metadata.csv', index=False)

        # Get the list of all the overview images
        overviews = df[df['Object']=='01']['ImgPath'].values

        # Load all of the overview images and display them
        imgs = []
        for o in overviews:
            img = tifffile.imread(o)
            imgs.append(img)
        imgs = np.array(imgs)
        self.viewer.layers.clear()
        self.viewer.add_image(imgs, name='overviews', channel_axis=1)
        self.viewer.layers[0].colormap = 'red'

        # Now draw the ROI boxes where the high res images are
        boxes = []
        sub_df = df[df['Object']!='01']
        for idx in range(0, len(sub_df)):
            tray_slide = 100*(int(sub_df.iloc[idx]['Tray'])-1) + int(sub_df.iloc[idx]['Slide']) - 1
            y = sub_df.iloc[idx]['Ix']
            x = sub_df.iloc[idx]['Iy']
            sdy = sub_df.iloc[idx]['sdx']
            sdx = sub_df.iloc[idx]['sdy']
            z = tray_slide
            box = np.array([[z, x, y], [z, x+sdx, y], [z, x+sdx, y+sdy], [z, x, y+sdy]])
            boxes.append(box)

        self.viewer.add_shapes(boxes)
        self.viewer.add_shapes([[[0,0,0], [0,0,1]]], shape_type='line')

        self.df = df

    def _on_save_highres(self):
        viewer = self.viewer

        df = self.df
        # Setup the resolution
        res_dict = {'Original':1, '2X':2, '4X':3, '8X':4}
        res_text = self.dropdown_menu.currentText()
        res = res_dict[res_text]

        lines = viewer.layers[-1].data
        pts_list = []
        for lin in lines:
            for a in range(0, len(lin)-1):
                point1 = lin[a]
                point2 = lin[a+1]
                points = make_points_along_line(point1, point2, number=30)
                pts_list = pts_list + points.tolist()

        pts_list = np.array(pts_list)
        positions = []
        #for pt in viewer.layers[-1].data:
        for pt in pts_list:
            df_pos = -1
            z = pt[0]
            rois = np.array(viewer.layers[-2].data)
            c_rois = rois[rois[:,:,0]==z].reshape(-1,4,3)[:,:,[1,2]]
            for a, rec in enumerate(c_rois):
                path = Path(rec)
                contained=path.contains_points([pt[1:3]])
                if np.any(contained):
                    df_pos = a
            positions.append([z,df_pos])
        positions=np.array(positions)
        positions = positions.astype(int)

        read_list = []
        for p in positions:
            mask = (df['TraySlide']==p[0]) & (df['Object']!='01')
            if p[1] != -1:
                read_list.append(df[mask].iloc[p[1]]['ImgPath'])

        read_list = pd.unique(read_list)

        max_y = np.floor(df[df['ImgPath'].isin(read_list)].max()['sx'] / (2**(res-1))).astype(int) + 1
        max_x = np.floor(df[df['ImgPath'].isin(read_list)].max()['sy'] / (2**(res-1))).astype(int) + 1

        imgs = []
        for f in read_list:
            img = tifffile.imread(f) + 1
            cx = img.shape[1]
            cy = img.shape[2]
            px = int(np.floor((max_x - cx)/2))
            py = int(np.floor((max_y - cy)/2))
            padded_img = np.pad(img, ((0,0),(px,max_x-cx-px),(py,max_y-cy-py)), 'constant')

            mask = padded_img==0
            noise = np.random.normal(219, 3, mask.shape)
            padded_img[np.where(mask)] = noise[np.where(mask)]

            imgs.append(padded_img)
        imgs = np.array(imgs)

        viewer.add_image(imgs, channel_axis=1)
        viewer.layers[-3].colormap = 'red'
        tifffile.imwrite(self.root_directory + '/Combined.tif', imgs.astype(np.ubyte), imagej=True, metadata={'axes':'ZCYX', 'mode':'color'})

class VSISelectXMLQWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # in one of two ways:
    # 1. use a parameter called `napari_viewer`, as done here
    # 2. use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer

        load_directory_btn = QPushButton("Choose folder")
        load_directory_btn.clicked.connect(self._on_choose_directory)
        export_vsi_btn = QPushButton("Export VSI")
        export_vsi_btn.clicked.connect(self._on_export_vsi)
        self.text_field = QLineEdit(self)
        self.r_channel_field = QLineEdit(self)
        self.g_channel_field = QLineEdit(self)
        self.b_channel_field = QLineEdit(self)
        self.r_channel_min = QLineEdit(self)
        self.r_channel_max = QLineEdit(self)
        self.g_channel_min = QLineEdit(self)
        self.g_channel_max = QLineEdit(self)
        self.b_channel_min = QLineEdit(self)
        self.b_channel_max = QLineEdit(self)

        self.r_channel_field.setText('1')
        self.g_channel_field.setText('2')
        self.b_channel_field.setText('3')
        self.r_channel_min.setText('0')
        self.r_channel_max.setText('20000')
        self.g_channel_min.setText('0')
        self.g_channel_max.setText('20000')
        self.b_channel_min.setText('0')
        self.b_channel_max.setText('20000')

        self.setLayout(QGridLayout())
        self.layout().addWidget(load_directory_btn, 0, 0)
        self.layout().addWidget(self.text_field, 0, 1)

        self.layout().addWidget(QLabel("Colors"), 1, 0)
        self.layout().addWidget(QLabel("Red"), 1, 1)
        self.layout().addWidget(QLabel("Green:"), 1, 2)
        self.layout().addWidget(QLabel("Blue:"), 1, 3)

        self.layout().addWidget(QLabel("Channel:"), 2, 0)
        self.layout().addWidget(self.r_channel_field, 2, 1)
        self.layout().addWidget(self.g_channel_field, 2, 2)
        self.layout().addWidget(self.b_channel_field, 2, 3)

        self.layout().addWidget(QLabel("Minimum:"), 3, 0)
        self.layout().addWidget(self.r_channel_min, 3, 1)
        self.layout().addWidget(self.g_channel_min, 3, 2)
        self.layout().addWidget(self.b_channel_min, 3, 3)

        self.layout().addWidget(QLabel("Maximum:"), 4, 0)
        self.layout().addWidget(self.r_channel_max, 4, 1)
        self.layout().addWidget(self.g_channel_max, 4, 2)
        self.layout().addWidget(self.b_channel_max, 4, 3)

        self.layout().addWidget(export_vsi_btn, 5, 1)

        self.layout().setSpacing(3)


        self.downsample = 4

    def _on_choose_directory(self):
        

        options = QFileDialog.Options()
        options |= QFileDialog.ReadOnly
        self.root_directory = QFileDialog.getExistingDirectory(self, "Select Directory", "", options=options)

        self.text_field.setText(self.root_directory)

        imgs, x, y = get_all_folders(self.root_directory)

        x_max = np.max(np.array([i.shape[1] for i in imgs]))
        y_max = np.max(np.array([i.shape[2] for i in imgs]))

        xdims = np.max(x) + x_max
        ydims = np.max(y) + y_max

        xdims = np.ceil(xdims/self.downsample).astype(int)
        ydims = np.ceil(ydims/self.downsample).astype(int)

        master_img = np.zeros((imgs[0].shape[0], xdims, ydims))
        for idx, img in enumerate(imgs):
            ds_img = img[:,::self.downsample,::self.downsample]
            xstart = np.floor(x[idx]/self.downsample).astype(int)
            ystart = np.floor(y[idx]/self.downsample).astype(int)
            xrng = ds_img.shape[1]
            yrng = ds_img.shape[2]
            master_img[:,xstart:xstart+xrng, ystart:ystart+yrng] = ds_img
        self.viewer.layers.clear()
        self.viewer.add_image(master_img, channel_axis=0, contrast_limits=[0,20000])

        # Make ROI boxes
        boxes = []
        for i, img in enumerate(imgs):
            x_min = int(x[i]/self.downsample)
            y_min = int(y[i]/self.downsample)
            x_max = int(img.shape[1]/self.downsample) + x_min
            y_max = int(img.shape[2]/self.downsample) + y_min
            boxes.append([[0, x_min, y_min], [0, x_max, y_min], [0, x_max, y_max], [0, x_min, y_max]])
        self.viewer.add_shapes(boxes, opacity=0.2)
        self.viewer.add_shapes([[[0,0,0], [0,0,1]]], shape_type='line', edge_width=10)

        self.imgs = imgs

    def _on_export_vsi(self):
        # Setup the resolution
        viewer = self.viewer
        lines = viewer.layers[-1].data
        read_list = get_order_of_images(lines, viewer)

        
        r = int(self.r_channel_field.text())
        g = int(self.g_channel_field.text())
        b = int(self.b_channel_field.text())
        print(read_list)

        if not os.path.exists(self.root_directory+'/outputs/'):
            # Create folder
            os.makedirs(self.root_directory+'/outputs/')

        for idx, i in enumerate(read_list):
            #viewer.add_image(self.imgs[i], channel_axis=0, contrast_limits=[0,20000])
            back_subbed = np.array([backsub(img.compute()) for img in self.imgs[i]])
            tifffile.imwrite(self.root_directory+'/outputs/'+'{:0>2}'.format(idx) + '.tif', back_subbed, imagej=True, metadata={'axes':'CYX', 'mode':'color'})

            ptiles = np.percentile(back_subbed, 99.99, axis=(1,2))
            #back_subbed_old = back_subbed.astype(float) * 255 / ptiles[:,np.newaxis, np.newaxis]

            back_subbed = back_subbed[[r,g,b],:,:]
            back_subbed[0] = clip_image(back_subbed[0], int(self.r_channel_min.text()), int(self.r_channel_max.text()))
            back_subbed[1] = clip_image(back_subbed[1], int(self.g_channel_min.text()), int(self.g_channel_max.text()))
            back_subbed[2] = clip_image(back_subbed[2], int(self.b_channel_min.text()), int(self.b_channel_max.text()))

            jpg = Image.fromarray(back_subbed.transpose(1,2,0).astype('uint8'))
            jpg.save(self.root_directory+'/outputs/'+'{:0>2}'.format(idx) + '.jpg')


    



