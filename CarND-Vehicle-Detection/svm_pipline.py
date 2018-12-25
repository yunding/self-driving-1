import matplotlib.image as mpimg
import cv2
import numpy as np
from skimage.feature import hog
from scipy.ndimage.measurements import label
from os import walk
from os import path
import time
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib
import os
import matplotlib.pyplot as plt
import multiprocessing
import pickle
from copy import copy
import glob
from timeit import default_timer as timer
from IPython.display import HTML
from moviepy.editor import VideoFileClip
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
import time
from multiprocessing import Pool

# 1 get features from image
# 2 standard features
# 3 use svm to train and predict whether the image contains vehicle
# 4 use slide window and find out heat area
# 5 draw boxes

# define parameters
orient = 11
pix_per_cell = 16
cell_per_block = 2
hog_channel = 'ALL'
color_space = 'YCrCb'
spatial_size = (32, 32)
hist_bins = 32
spatial_feat = True
hist_feat = True
hog_feat = True
svc = joblib.load("model2/train_model.m")
X_scaler = joblib.load("model2/train_scaler.m")


# Define a function to compute binned color features
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features


# Define a function to compute color histogram features
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features


# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist(),get_hog
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # 1) Define an empty list to receive features
    img_features = []
    # 2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else:
        feature_image = np.copy(img)
    # 3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        # 4) Append features to list
        img_features.append(spatial_features)
    # 5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        # 6) Append features to list
        img_features.append(hist_features)
    # 7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                     orient, pix_per_cell, cell_per_block,
                                                     vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                            pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        # 8) Append features to list
        img_features.append(hog_features)

    # 9) Return concatenated array of features
    return np.concatenate(img_features)


def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                     hist_bins=32, orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images

    for file in imgs:
        file_features = []
        # Read in each one by one
        try:
            image = cv2.imread(file)
        except:
            print(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else:
            feature_image = np.copy(image)

        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.append(get_hog_features(feature_image[:, :, channel],
                                                         orient, pix_per_cell, cell_per_block,
                                                         vis=False, feature_vec=True))
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(feature_image[:, :, hog_channel], orient,
                                                pix_per_cell, cell_per_block, vis=False, feature_vec=True)
            # Append the new feature vector to the features list
            file_features.append(hog_features)
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features


def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.75, 0.75)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0] * (1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1] * (1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_buffer = np.int(xy_window[0] * (xy_overlap[0]))
    ny_buffer = np.int(xy_window[1] * (xy_overlap[1]))
    nx_windows = np.int((xspan - nx_buffer) / nx_pix_per_step)
    ny_windows = np.int((yspan - ny_buffer) / ny_pix_per_step)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs * nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys * ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]
            # Append window position to list
            if starty > 400 and endy <= 600:
                window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list


def search_windows(img, windows, clf, scaler, color_space='RGB',
                   spatial_size=(32, 32), hist_bins=32,
                   hist_range=(0, 256), orient=9,
                   pix_per_cell=8, cell_per_block=2,
                   hog_channel=0, spatial_feat=True,
                   hist_feat=True, hog_feat=True):
    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()

        features = single_img_features(test_img, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)
        # 5) Scale extracted features to be fed to classifier
        # test_features = StandardScaler().fit_transform(features)
        X = np.hstack(features).reshape(1, -1)
        test_features = scaler.transform(X)
        # test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows


def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap  # Iterate through list of bboxes


def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap


def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1] + 1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return img


def get_file_name(rootDir):
    import os
    data = []
    for root, dirs, files in os.walk(rootDir, topdown=False):
        for name in files:
            _, ending = path.splitext(name)
            if ending != ".jpg" and ending != ".jepg" and ending != ".png":
                continue
            else:
                data.append(path.join(root, name))
    return data


# train model2 with the classfied images

def train_model():
    cars = []
    not_cars = []
    vehicles = get_file_name("data\\vehicles")
    not_vehicles = get_file_name("data\\non-vehicles")
    sample_size = min(len(not_vehicles), len(vehicles))
    cars = vehicles[0:sample_size]
    not_cars = not_vehicles[0:sample_size]
    # for i in range(0, sample_size):
    #     if i % 5 == 0:
    #         cars.append(vehicles[i])
    #         not_cars.append(not_vehicles[i])
    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    not_features = extract_features(not_cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    x_features = np.vstack((car_features, not_features)).astype(np.float64)

    x_scaler = StandardScaler().fit(x_features)
    x = x_scaler.transform(x_features)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(not_features))))
    rate = 42
    train_data, test_data, train_label, test_label = train_test_split(x, y, test_size=0.33, random_state=rate)
    # parameters = {'C': {0.1, 1, 10}}
    # parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
    #                'C': [1, 10, 100, 1000]}
    scores = ['precision', 'recall']
    clf = LinearSVC()
    # #svr = svc()
    # train_predict(scores, parameters, train_data, train_label, test_data, test_label)
    # clf = GridSearchCV(svr, parameters, score = scores)
    clf.fit(train_data, train_label)
    score = clf.score(test_data, test_label)
    print("train model2 score:", score)
    joblib.dump(clf, "model2/train_model.m")
    joblib.dump(x_scaler, "model2/train_scaler.m")


# recover
# clf = joblib.load("model2/train_model.m")


def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def find_windows(image, window_size
                 , svc, X_scaler,color_space,spatial_size,hist_bins,orient,
                 pix_per_cell,cell_per_block,hog_channel,spatial_feat,hist_feat,hog_feat):
    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                           xy_window=(window_size, window_size), xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)
    return hot_windows

last_windows = []
def find_car(image, svc=svc, X_scaler=X_scaler):
    global last_windows
    window_size = [64, 96]
    result1 = []
    draw_windows = []
    p = Pool()
    for size in window_size:
        result1.append(p.apply_async(find_windows, args=(image, size, svc, X_scaler,
                                                        color_space, spatial_size, hist_bins, orient,
                                                        pix_per_cell, cell_per_block, hog_channel, spatial_feat,
                                                        hist_feat, hog_feat
                                                )))
    p.close()
    p.join()
    for res in result1:
        draw_windows.extend(res.get())
    if len(last_windows) <= 0:
        last_windows = draw_windows
    true_windows = list(set(last_windows).intersection(set(draw_windows)))
    heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    heat = add_heat(heat, true_windows)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 5)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = draw_labeled_bboxes(np.copy(image), labels)

    return draw_img, heat, heatmap



def process(image):
    draw_img, heat, heatmap = find_car(image)
    return draw_img


if __name__ == '__main__':
    #train_model()
    # video_output = "video_out/project_video_output.mp4"
    # clip1 = VideoFileClip("project_video.mp4")
    # i = 0
    # for frame in clip1.iter_frames():
    #     if i < 1260:
    #         plt.imshow(clip1.get_frame(i+1))
    #     i = i + 1
    #     process(im)
    # clip1_output = clip1.fl_image(process)  # NOTE: this function expects color images!!
    # clip1_output.write_videofile(video_output, audio=False)
    #train_model()
    image_paths = glob.glob("D:\\code\\python\\self-driving\\CarND-Vehicle-Detection\\test_images\\test6.jpg")
    for path in image_paths:
       name = path.split("\\")[6]
       img= mpimg.imread(path)
       draw_img=process(img)
       plt.imshow(draw_img)
       plt.show()
       print("ok")

    # image1 = mpimg.imread("D:\\code\\python\\self-driving\\CarND-Vehicle-Detection\\test_images\\test5.jpg")
    # image2 = mpimg.imread("D:\\code\\python\\self-driving\\CarND-Vehicle-Detection\\test_images\\test4.jpg")
    # ystart = 400
    # ystop = 656
    # scale = 1.5
    # colorspace = 'YUV'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    # orient = 11
    # pix_per_cell = 16
    # cell_per_block = 2
    # hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
    #
    # rectangles = find_cars(image1, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell,cell_per_block, spatial_size,hist_bins)
    # plt.title('draw images')
    # image = cv2.imread("test6.png")
    # draw_image = find_car(image, svc, X_scaler)
    # windows = slide_window(image, xy_window=(64, 64), xy_overlap=(0.75, 0.75))
    # hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
    #                              spatial_size=spatial_size, hist_bins=hist_bins,
    #                              orient=orient, pix_per_cell=pix_per_cell,
    #                              cell_per_block=cell_per_block,
    #                              hog_channel=hog_channel, spatial_feat=spatial_feat,
    #                              hist_feat=hist_feat, hog_feat=hog_feat)
    # img = draw_boxes(image, hot_windows)
    # heat = np.zeros_like(image[:, :, 0]).astype(np.float)
    # heat = add_heat(heat, hot_windows)
    # threshold = apply_threshold(heat, 0)
    # plt.imshow(threshold)
    # plt.show()
    # print("ok")
