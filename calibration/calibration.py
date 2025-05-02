import cv2 as cv
import glob
import numpy as np
import sys
from scipy import linalg
import yaml
import os

# python calibrate.py calibration_settings.yaml
#This will contain the calibration settings from the calibration_settings.yaml file
calibration_settings = {}

#Given Projection matrices P1 and P2, and pixel coordinates point1 and point2, return triangulated 3D point.
def DLT(P1, P2, point1, point2):

    A = [point1[1]*P1[2,:] - P1[1,:],
         P1[0,:] - point1[0]*P1[2,:],
         point2[1]*P2[2,:] - P2[1,:],
         P2[0,:] - point2[0]*P2[2,:]
        ]
    A = np.array(A).reshape((4,4))

    B = A.transpose() @ A
    U, s, Vh = linalg.svd(B, full_matrices = False)

    #print('Triangulated point: ')
    #print(Vh[3,0:3]/Vh[3,3])
    return Vh[3,0:3]/Vh[3,3]


#Open and load the calibration_settings.yaml file
def parse_calibration_settings_file(filename):
    
    global calibration_settings

    if not os.path.exists(filename):
        print('File does not exist:', filename)
        quit()
    
    print('Using for calibration settings: ', filename)

    with open(filename) as f:
        calibration_settings = yaml.safe_load(f)

    #rudimentray check to make sure correct file was loaded
    if 'camera0' not in calibration_settings.keys():
        print('camera0 key was not found in the settings file. Check if correct calibration_settings.yaml file was passed')
        quit()


def save_frames_single_camera(camera_name):

    # create frames directory
    if not os.path.exists('frames'):
        os.mkdir('frames')

    # get settings
    camera_device_id = calibration_settings[camera_name]
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    number_to_save = calibration_settings['mono_calibration_frames']
    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']

    # open video stream and change resolution.
    cap = cv.VideoCapture(camera_device_id)
    cap.set(3, width * 2)  # Tek lens görüntüsü için genişliği iki katı yapıyoruz.
    cap.set(4, height)
    
    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if ret == False:
            print("No video data received from camera. Exiting...")
            quit()

        # Sol ve sağ lens görüntülerini ayır
        left_frame = frame[:, :frame.shape[1] // 2]  # Sol lens
        frame_small = cv.resize(left_frame, None, fx=1/view_resize, fy=1/view_resize)

        if not start:
            cv.putText(frame_small, "Press SPACEBAR to start collection frames", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame_small, "Cooldown: " + str(cooldown), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame_small, "Num frames: " + str(saved_count), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            # save the frame when cooldown reaches 0.
            if cooldown <= 0:
                savename = os.path.join('frames', camera_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename, left_frame)  # Sol lens kaydediliyor
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame_small', frame_small)
        k = cv.waitKey(1)

        if k == 27:
            quit()

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()


#Calibrate single camera to obtain camera intrinsic parameters from saved frames.
def calibrate_camera_for_intrinsic_parameters(images_prefix):
    
    #NOTE: images_prefix contains camera name: "frames/camera0*".
    images_names = glob.glob(images_prefix)

    #read all frames
    images = [cv.imread(imname, 1) for imname in images_names]

    #criteria used by checkerboard pattern detector.
    #Change this if the code can't find the checkerboard. 
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale'] #this will change to user defined length scale

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = images[0].shape[1]
    height = images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints = [] # 2d points in image plane.

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space


    for i, frame in enumerate(images):
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        #find the checkerboard
        ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

        if ret == True:

            #Convolution size used to improve corner detection. Don't make this too large.
            conv_size = (11, 11)

            #opencv can attempt to improve the checkerboard coordinates
            corners = cv.cornerSubPix(gray, corners, conv_size, (-1, -1), criteria)
            cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
            cv.putText(frame, 'If detected points are poor, press "s" to skip this sample', (25, 25), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)

            cv.imshow('img', frame)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints.append(corners)


    cv.destroyAllWindows()
    ret, cmtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print('rmse:', ret)
    print('camera matrix:\n', cmtx)
    print('distortion coeffs:', dist)

    return cmtx, dist

#save camera intrinsic parameters to file
def save_camera_intrinsics(camera_matrix, distortion_coefs, camera_name):

    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    out_filename = os.path.join('camera_parameters', camera_name + '_intrinsics.dat')
    outf = open(out_filename, 'w')

    outf.write('intrinsic:\n')
    for l in camera_matrix:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('distortion:\n')
    for en in distortion_coefs[0]:
        outf.write(str(en) + ' ')
    outf.write('\n')


def save_frames_two_cams(camera0_name, camera1_name):

    if not os.path.exists('frames_pair'):
        os.mkdir('frames_pair')

    view_resize = calibration_settings['view_resize']
    cooldown_time = calibration_settings['cooldown']
    number_to_save = calibration_settings['stereo_calibration_frames']

    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    cap0.set(3, calibration_settings['frame_width'] * 2)
    cap0.set(4, calibration_settings['frame_height'])
    cap1.set(3, calibration_settings['frame_width'] * 2)
    cap1.set(4, calibration_settings['frame_height'])

    cooldown = cooldown_time
    start = False
    saved_count = 0

    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Cameras not returning video data. Exiting...')
            quit()

        # Sadece sol lens görüntülerini al
        left_frame0 = frame0[:, :frame0.shape[1] // 2]
        left_frame1 = frame1[:, :frame1.shape[1] // 2]

        frame0_small = cv.resize(left_frame0, None, fx=1./view_resize, fy=1./view_resize)
        frame1_small = cv.resize(left_frame1, None, fx=1./view_resize, fy=1./view_resize)

        if not start:
            cv.putText(frame0_small, "Make sure both cameras can see the calibration pattern well", (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
            cv.putText(frame0_small, "Press SPACEBAR to start collection frames", (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

        if start:
            cooldown -= 1
            cv.putText(frame0_small, "Cooldown: " + str(cooldown), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame0_small, "Num frames: " + str(saved_count), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            cv.putText(frame1_small, "Cooldown: " + str(cooldown), (50, 50), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
            cv.putText(frame1_small, "Num frames: " + str(saved_count), (50, 100), cv.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)

            if cooldown <= 0:
                savename0 = os.path.join('frames_pair', camera0_name + '_' + str(saved_count) + '.png')
                savename1 = os.path.join('frames_pair', camera1_name + '_' + str(saved_count) + '.png')
                cv.imwrite(savename0, left_frame0)
                cv.imwrite(savename1, left_frame1)
                saved_count += 1
                cooldown = cooldown_time

        cv.imshow('frame0_small', frame0_small)
        cv.imshow('frame1_small', frame1_small)
        k = cv.waitKey(1)

        if k == 27:
            quit()

        if k == 32:
            start = True

        if saved_count == number_to_save:
            break

    cv.destroyAllWindows()



#open paired calibration frames and stereo calibrate for cam0 to cam1 coorindate transformations
def stereo_calibrate(mtx0, dist0, mtx1, dist1, frames_prefix_c0, frames_prefix_c1):
    #read the synched frames
    c0_images_names = sorted(glob.glob(frames_prefix_c0))
    c1_images_names = sorted(glob.glob(frames_prefix_c1))

    #open images
    c0_images = [cv.imread(imname, 1) for imname in c0_images_names]
    c1_images = [cv.imread(imname, 1) for imname in c1_images_names]

    #change this if stereo calibration not good.
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.001)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    #frame dimensions. Frames should be the same size.
    width = c0_images[0].shape[1]
    height = c0_images[0].shape[0]

    #Pixel coordinates of checkerboards
    imgpoints_left = [] # 2d points in image plane.
    imgpoints_right = []

    #coordinates of the checkerboard in checkerboard world space.
    objpoints = [] # 3d point in real world space

    for frame0, frame1 in zip(c0_images, c1_images):
        gray1 = cv.cvtColor(frame0, cv.COLOR_BGR2GRAY)
        gray2 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv.findChessboardCorners(gray1, (rows, columns), None)
        c_ret2, corners2 = cv.findChessboardCorners(gray2, (rows, columns), None)

        if c_ret1 == True and c_ret2 == True:

            corners1 = cv.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            p0_c1 = corners1[0,0].astype(np.int32)
            p0_c2 = corners2[0,0].astype(np.int32)

            cv.putText(frame0, 'O', (p0_c1[0], p0_c1[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame0, (rows,columns), corners1, c_ret1)
            cv.imshow('img', frame0)

            cv.putText(frame1, 'O', (p0_c2[0], p0_c2[1]), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
            cv.drawChessboardCorners(frame1, (rows,columns), corners2, c_ret2)
            cv.imshow('img2', frame1)
            k = cv.waitKey(0)

            if k & 0xFF == ord('s'):
                print('skipping')
                continue

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    stereocalibration_flags = cv.CALIB_FIX_INTRINSIC
    ret, CM1, dist0, CM2, dist1, R, T, E, F = cv.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx0, dist0,
                                                                 mtx1, dist1, (width, height), criteria = criteria, flags = stereocalibration_flags)

    print('rmse: ', ret)
    cv.destroyAllWindows()
    return R, T

#Converts Rotation matrix R and Translation vector T into a homogeneous representation matrix
def _make_homogeneous_rep_matrix(R, t):
    P = np.zeros((4,4))
    P[:3,:3] = R
    P[:3, 3] = t.reshape(3)
    P[3,3] = 1
 
    return P
# Turn camera calibration data into projection matrix
def get_projection_matrix(cmtx, R, T):
    P = cmtx @ _make_homogeneous_rep_matrix(R, T)[:3,:]
    return P

# After calibrating, we can see shifted coordinate axes in the video feeds directly
def check_calibration(camera0_name, camera0_data, camera1_name, camera1_data, _zshift = 50.):
    
    cmtx0 = np.array(camera0_data[0])
    dist0 = np.array(camera0_data[1])
    R0 = np.array(camera0_data[2])
    T0 = np.array(camera0_data[3])
    cmtx1 = np.array(camera1_data[0])
    dist1 = np.array(camera1_data[1])
    R1 = np.array(camera1_data[2])
    T1 = np.array(camera1_data[3])

    P0 = get_projection_matrix(cmtx0, R0, T0)
    P1 = get_projection_matrix(cmtx1, R1, T1)

    #define coordinate axes in 3D space. These are just the usual coorindate vectors
    coordinate_points = np.array([[0.,0.,0.],
                                  [1.,0.,0.],
                                  [0.,1.,0.],
                                  [0.,0.,1.]])
    z_shift = np.array([0.,0.,_zshift]).reshape((1, 3))
    #increase the size of the coorindate axes and shift in the z direction
    draw_axes_points = 5 * coordinate_points + z_shift

    #project 3D points to each camera view manually. This can also be done using cv.projectPoints()
    #Note that this uses homogenous coordinate formulation
    pixel_points_camera0 = []
    pixel_points_camera1 = []
    for _p in draw_axes_points:
        X = np.array([_p[0], _p[1], _p[2], 1.])
        
        #project to camera0
        uv = P0 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera0.append(uv)

        #project to camera1
        uv = P1 @ X
        uv = np.array([uv[0], uv[1]])/uv[2]
        pixel_points_camera1.append(uv)

    #these contain the pixel coorindates in each camera view as: (pxl_x, pxl_y)
    pixel_points_camera0 = np.array(pixel_points_camera0)
    pixel_points_camera1 = np.array(pixel_points_camera1)

    #open the video streams
    cap0 = cv.VideoCapture(calibration_settings[camera0_name])
    cap1 = cv.VideoCapture(calibration_settings[camera1_name])

    #set camera resolutions
    width = calibration_settings['frame_width']
    height = calibration_settings['frame_height']
    cap0.set(3, width)
    cap0.set(4, height)
    cap1.set(3, width)
    cap1.set(4, height)

    while True:

        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print('Video stream not returning frame data')
            quit()

        #follow RGB colors to indicate XYZ axes respectively
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        #draw projections to camera0
        origin = tuple(pixel_points_camera0[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera0[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame0, origin, _p, col, 2)
        
        #draw projections to camera1
        origin = tuple(pixel_points_camera1[0].astype(np.int32))
        for col, _p in zip(colors, pixel_points_camera1[1:]):
            _p = tuple(_p.astype(np.int32))
            cv.line(frame1, origin, _p, col, 2)

        cv.imshow('frame0', frame0)
        cv.imshow('frame1', frame1)

        k = cv.waitKey(1)
        if k == 27: break

    cv.destroyAllWindows()

def get_world_space_origin(cmtx, dist, img_path):

    frame = cv.imread(img_path, 1)

    #calibration pattern settings
    rows = calibration_settings['checkerboard_rows']
    columns = calibration_settings['checkerboard_columns']
    world_scaling = calibration_settings['checkerboard_box_size_scale']

    #coordinates of squares in the checkerboard world space
    objp = np.zeros((rows*columns,3), np.float32)
    objp[:,:2] = np.mgrid[0:rows,0:columns].T.reshape(-1,2)
    objp = world_scaling* objp

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (rows, columns), None)

    cv.drawChessboardCorners(frame, (rows,columns), corners, ret)
    cv.putText(frame, "If you don't see detected points, try with a different image", (50,50), cv.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 1)
    cv.imshow('img', frame)
    cv.waitKey(0)

    ret, rvec, tvec = cv.solvePnP(objp, corners, cmtx, dist)
    R, _  = cv.Rodrigues(rvec) #rvec is Rotation matrix in Rodrigues vector form

    return R, tvec

def get_cam1_to_world_transforms(cmtx0, dist0, R_W0, T_W0, 
                                 cmtx1, dist1, R_01, T_01,
                                 image_path0,
                                 image_path1):

    frame0 = cv.imread(image_path0, 1)
    frame1 = cv.imread(image_path1, 1)

    unitv_points = 5 * np.array([[0,0,0], [1,0,0], [0,1,0], [0,0,1]], dtype = 'float32').reshape((4,1,3))
    #axes colors are RGB format to indicate XYZ axes.
    colors = [(0,0,255), (0,255,0), (255,0,0)]

    #project origin points to frame 0
    points, _ = cv.projectPoints(unitv_points, R_W0, T_W0, cmtx0, dist0)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame0, origin, _p, col, 2)

    #project origin points to frame1
    R_W1 = R_01 @ R_W0
    T_W1 = R_01 @ T_W0 + T_01
    points, _ = cv.projectPoints(unitv_points, R_W1, T_W1, cmtx1, dist1)
    points = points.reshape((4,2)).astype(np.int32)
    origin = tuple(points[0])
    for col, _p in zip(colors, points[1:]):
        _p = tuple(_p.astype(np.int32))
        cv.line(frame1, origin, _p, col, 2)

    cv.imshow('frame0', frame0)
    cv.imshow('frame1', frame1)
    cv.waitKey(0)

    return R_W1, T_W1


def save_extrinsic_calibration_parameters(R0, T0, R1, T1, prefix = ''):
    
    #create folder if it does not exist
    if not os.path.exists('camera_parameters'):
        os.mkdir('camera_parameters')

    camera0_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera0_rot_trans.dat')
    outf = open(camera0_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T0:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    #R1 and T1 are just stereo calibration returned values
    camera1_rot_trans_filename = os.path.join('camera_parameters', prefix + 'camera1_rot_trans.dat')
    outf = open(camera1_rot_trans_filename, 'w')

    outf.write('R:\n')
    for l in R1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')

    outf.write('T:\n')
    for l in T1:
        for en in l:
            outf.write(str(en) + ' ')
        outf.write('\n')
    outf.close()

    return R0, T0, R1, T1

if __name__ == '__main__':

    if len(sys.argv) != 2:
        print('Call with settings filename: "python3 calibrate.py calibration_settings.yaml"')
        quit()
    
    #Open and parse the settings file
    parse_calibration_settings_file(sys.argv[1])


    """Step1. Save calibration frames for single cameras"""
    save_frames_single_camera('camera0') #save frames for camera0
    save_frames_single_camera('camera1') #save frames for camera1


    """Step2. Obtain camera intrinsic matrices and save them"""
    #camera0 intrinsics
    images_prefix = os.path.join('frames', 'camera0*')
    cmtx0, dist0 = calibrate_camera_for_intrinsic_parameters(images_prefix) 
    save_camera_intrinsics(cmtx0, dist0, 'camera0') #this will write cmtx and dist to disk
    #camera1 intrinsics
    images_prefix = os.path.join('frames', 'camera1*')
    cmtx1, dist1 = calibrate_camera_for_intrinsic_parameters(images_prefix)
    save_camera_intrinsics(cmtx1, dist1, 'camera1') #this will write cmtx and dist to disk


    """Step3. Save calibration frames for both cameras simultaneously"""
    save_frames_two_cams('camera0', 'camera1') #save simultaneous frames


    """Step4. Use paired calibration pattern frames to obtain camera0 to camera1 rotation and translation"""
    frames_prefix_c0 = os.path.join('frames_pair', 'camera0*')
    frames_prefix_c1 = os.path.join('frames_pair', 'camera1*')
    R, T = stereo_calibrate(cmtx0, dist0, cmtx1, dist1, frames_prefix_c0, frames_prefix_c1)


    """Step5. Save calibration data where camera0 defines the world space origin."""
    #camera0 rotation and translation is identity matrix and zeros vector
    R0 = np.eye(3, dtype=np.float32)
    T0 = np.array([0., 0., 0.]).reshape((3, 1))

    save_extrinsic_calibration_parameters(R0, T0, R, T) #this will write R and T to disk
    R1 = R; T1 = T #to avoid confusion, camera1 R and T are labeled R1 and T1
    #check your calibration makes sense
    camera0_data = [cmtx0, dist0, R0, T0]
    camera1_data = [cmtx1, dist1, R1, T1]
    check_calibration('camera0', camera0_data, 'camera1', camera1_data, _zshift = 60.)#!/usr/bin/env python3
"""
Stereo-Camera Calibration Utility
=================================

Captures calibration images, computes intrinsic and extrinsic parameters
for two fisheye/dual-lens USB cameras and verifies the result visually.

Usage
-----
$ python calibrate.py calibration_settings.yaml
"""

from __future__ import annotations

import os
import sys
import glob
import yaml
import cv2 as cv
import numpy as np
from typing import Dict, List, Tuple
from scipy import linalg

# -----------------------------------------------------------------------------#
# Global configuration (loaded from YAML)
# -----------------------------------------------------------------------------#
calibration_settings: Dict[str, any] = {}


# -----------------------------------------------------------------------------#
# I/O helpers
# -----------------------------------------------------------------------------#
def _ensure_dir(path: str) -> None:
    """Create *path* directory if it does not already exist."""
    os.makedirs(path, exist_ok=True)


def _write_matrix(f, label: str, mat: np.ndarray) -> None:
    """Write a matrix to an open file, one row per line."""
    f.write(f"{label}:\n")
    for row in mat:
        f.write(" ".join(map(str, row)) + "\n")


# -----------------------------------------------------------------------------#
# Configuration
# -----------------------------------------------------------------------------#
def load_settings(file_name: str) -> None:
    """
    Load calibration parameters from a YAML file into *calibration_settings*.
    Exits with an error message if mandatory keys are missing.
    """
    global calibration_settings

    if not os.path.isfile(file_name):
        sys.exit(f"[ERROR] Settings file not found: {file_name}")

    with open(file_name, "r", encoding="utf-8") as fh:
        calibration_settings = yaml.safe_load(fh)

    # Rudimentary sanity check
    mandatory_keys = {"camera0", "camera1", "frame_width", "frame_height"}
    if not mandatory_keys.issubset(calibration_settings):
        sys.exit("[ERROR] Missing keys in YAML. Required: " + ", ".join(mandatory_keys))

    print(f"[INFO] Loaded settings from '{file_name}'")


# -----------------------------------------------------------------------------#
# Geometry utilities
# -----------------------------------------------------------------------------#
def dlt_triangulate(
    P1: np.ndarray, P2: np.ndarray, p1: np.ndarray, p2: np.ndarray
) -> np.ndarray:
    """
    Triangulate a single 3-D point from its projections *p1*, *p2*
    using Direct Linear Transformation (DLT).

    Returns
    -------
    np.ndarray, shape (3,)
        Cartesian coordinates in homogeneous space (X / W, Y / W, Z / W).
    """
    A = np.array(
        [
            p1[1] * P1[2, :] - P1[1, :],
            P1[0, :] - p1[0] * P1[2, :],
            p2[1] * P2[2, :] - P2[1, :],
            P2[0, :] - p2[0] * P2[2, :],
        ]
    )

    _, _, Vh = linalg.svd(A.T @ A, full_matrices=False)
    return (Vh[3, 0:3] / Vh[3, 3]).astype(np.float32)


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Return a 4×4 homogeneous transform from rotation *R* and translation *t*."""
    H = np.eye(4, dtype=R.dtype)
    H[:3, :3] = R
    H[:3, 3] = t.ravel()
    return H


def projection_matrix(K: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Compute 3×4 camera projection matrix P = K · [R | t]."""
    return K @ make_homogeneous(R, t)[:3, :]


# -----------------------------------------------------------------------------#
# Image capture
# -----------------------------------------------------------------------------#
def _resize_preview(frame: np.ndarray, scale: float) -> np.ndarray:
    """Return a resized copy of *frame* for fast live preview."""
    return cv.resize(frame, None, fx=1 / scale, fy=1 / scale)


def capture_single_camera(camera_key: str) -> None:
    """
    Grab checkerboard frames from one dual-lens camera (left half).
    Saves PNGs into ./frames/.
    """
    _ensure_dir("frames")

    cam_id = calibration_settings[camera_key]
    w, h = calibration_settings["frame_width"], calibration_settings["frame_height"]
    n_frames = calibration_settings["mono_calibration_frames"]
    scale = calibration_settings["view_resize"]
    cooldown_default = calibration_settings["cooldown"]

    cap = cv.VideoCapture(cam_id)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, w * 2)  # full stereo image
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    saved, cooldown, recording = 0, cooldown_default, False

    while saved < n_frames:
        ok, frame = cap.read()
        if not ok:
            sys.exit("[ERROR] No data from camera")

        left = frame[:, : frame.shape[1] // 2]
        preview = _resize_preview(left, scale)

        # UI overlay
        msg = (
            "Press SPACE to start" if not recording else
            f"Cooldown: {cooldown:2d}  |  Saved: {saved}/{n_frames}"
        )
        cv.putText(
            preview, msg, (40, 40), cv.FONT_HERSHEY_COMPLEX, 1,
            (0, 255, 0) if recording else (0, 0, 255), 2
        )

        cv.imshow(f"Preview - {camera_key}", preview)
        key = cv.waitKey(1) & 0xFF

        if key == 27:  # ESC
            sys.exit("[ABORT] User exit")
        if key == 32:  # SPACE
            recording = True

        if recording:
            cooldown -= 1
            if cooldown <= 0:
                filename = os.path.join("frames", f"{camera_key}_{saved:02d}.png")
                cv.imwrite(filename, left)
                print(f"[IMG] {filename}")
                saved += 1
                cooldown = cooldown_default

    cv.destroyAllWindows()


def capture_stereo_pair(cam0: str, cam1: str) -> None:
    """
    Capture synchronized checkerboard frames from *cam0* and *cam1*.
    Saves images to ./frames_pair/.
    """
    _ensure_dir("frames_pair")

    w, h = calibration_settings["frame_width"], calibration_settings["frame_height"]
    n_frames = calibration_settings["stereo_calibration_frames"]
    scale = calibration_settings["view_resize"]
    cooldown_default = calibration_settings["cooldown"]

    cap0 = cv.VideoCapture(calibration_settings[cam0])
    cap1 = cv.VideoCapture(calibration_settings[cam1])
    for cap in (cap0, cap1):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, w * 2)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, h)

    saved, cooldown, recording = 0, cooldown_default, False

    while saved < n_frames:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            sys.exit("[ERROR] Cameras disconnected")

        left0, left1 = f0[:, : f0.shape[1] // 2], f1[:, : f1.shape[1] // 2]
        p0 = _resize_preview(left0, scale)
        p1 = _resize_preview(left1, scale)

        msg = (
            "SPACE to start" if not recording else
            f"Cooldown: {cooldown:2d}  |  Saved: {saved}/{n_frames}"
        )
        for canvas in (p0, p1):
            cv.putText(canvas, msg, (30, 30), cv.FONT_HERSHEY_COMPLEX, 1,
                       (0, 255, 0) if recording else (0, 0, 255), 2)

        cv.imshow("Cam0", p0)
        cv.imshow("Cam1", p1)
        key = cv.waitKey(1) & 0xFF

        if key == 27:
            sys.exit("[ABORT] User exit")
        if key == 32:
            recording = True

        if recording:
            cooldown -= 1
            if cooldown <= 0:
                fn0 = os.path.join("frames_pair", f"{cam0}_{saved:02d}.png")
                fn1 = os.path.join("frames_pair", f"{cam1}_{saved:02d}.png")
                cv.imwrite(fn0, left0)
                cv.imwrite(fn1, left1)
                print(f"[IMG] {fn0}  |  {fn1}")
                saved += 1
                cooldown = cooldown_default

    cv.destroyAllWindows()


# -----------------------------------------------------------------------------#
# Calibration helpers
# -----------------------------------------------------------------------------#
def _generate_object_points() -> np.ndarray:
    """
    Create the canonical (x, y, 0) grid for the checkerboard pattern
    and scale it by *checkerboard_box_size_scale*.
    """
    rows = calibration_settings["checkerboard_rows"]
    cols = calibration_settings["checkerboard_columns"]
    scale = calibration_settings["checkerboard_box_size_scale"]

    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:rows, 0:cols].T.reshape(-1, 2)
    return objp * scale


def calibrate_intrinsics(img_pattern: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimate *K* (3×3) and distortion coefficients *dist* from
    all images matching *img_pattern* (e.g. frames/camera0*).
    """
    images = sorted(glob.glob(img_pattern))
    if not images:
        sys.exit(f"[ERROR] No images found for pattern {img_pattern}")

    objp = _generate_object_points()
    objpoints, imgpoints = [], []

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)
    preview = None

    for fname in images:
        frame = cv.imread(fname)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        ok, corners = cv.findChessboardCorners(gray, objp.shape[:2], None)

        if ok:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners)
            objpoints.append(objp)

            # Show detection for manual approval
            preview = frame.copy()
            cv.drawChessboardCorners(preview, objp.shape[:2], corners, ok)
            cv.putText(preview, "Press 's' to skip this sample",
                       (20, 30), cv.FONT_HERSHEY_PLAIN, 1.2, (0, 0, 255), 1)
            cv.imshow("Check", preview)
            if cv.waitKey(0) & 0xFF == ord("s"):
                objpoints.pop()
                imgpoints.pop()

    cv.destroyAllWindows()

    h, w = cv.imread(images[0]).shape[:2]
    rms, K, dist, *_ = cv.calibrateCamera(objpoints, imgpoints, (w, h), None, None)
    print(f"[CALIB] {img_pattern}: RMS = {rms:.4f}")
    print("[CALIB] K =\n", K)
    print("[CALIB] dist =", dist.ravel())
    return K, dist


def save_intrinsics(K: np.ndarray, dist: np.ndarray, cam_key: str) -> None:
    """Write intrinsic parameters to ./camera_parameters/."""
    _ensure_dir("camera_parameters")
    with open(f"camera_parameters/{cam_key}_intrinsics.dat", "w") as fh:
        _write_matrix(fh, "intrinsic", K)
        _write_matrix(fh, "distortion", dist)


def stereo_calibrate(
    K0: np.ndarray, d0: np.ndarray,
    K1: np.ndarray, d1: np.ndarray,
    pattern0: str, pattern1: str
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Derive rotation *R* (3×3) and translation *T* (3×1) that map camera0
    coordinates into camera1 coordinates.
    """
    imgs0 = sorted(glob.glob(pattern0))
    imgs1 = sorted(glob.glob(pattern1))
    if not (imgs0 and imgs1 and len(imgs0) == len(imgs1)):
        sys.exit("[ERROR] Stereo frame pairs missing or unsynchronized")

    objp = _generate_object_points()
    objpoints, imgpts_l, imgpts_r = [], [], []

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-3)

    for f0, f1 in zip(imgs0, imgs1):
        l_img, r_img = cv.imread(f0), cv.imread(f1)
        g0, g1 = cv.cvtColor(l_img, cv.COLOR_BGR2GRAY), cv.cvtColor(r_img, cv.COLOR_BGR2GRAY)

        ok0, c0 = cv.findChessboardCorners(g0, objp.shape[:2], None)
        ok1, c1 = cv.findChessboardCorners(g1, objp.shape[:2], None)
        if not (ok0 and ok1):
            continue

        c0 = cv.cornerSubPix(g0, c0, (11, 11), (-1, -1), criteria)
        c1 = cv.cornerSubPix(g1, c1, (11, 11), (-1, -1), criteria)

        objpoints.append(objp)
        imgpts_l.append(c0)
        imgpts_r.append(c1)

    h, w = cv.imread(imgs0[0]).shape[:2]
    flags = cv.CALIB_FIX_INTRINSIC
    rms, *_ , R, T, _, _ = cv.stereoCalibrate(
        objpoints, imgpts_l, imgpts_r,
        K0, d0, K1, d1, (w, h),
        criteria=criteria, flags=flags
    )

    print(f"[STEREO] RMS = {rms:.4f}")
    print("[STEREO] R =\n", R)
    print("[STEREO] T =\n", T.ravel())
    return R, T


def save_extrinsics(R0: np.ndarray, t0: np.ndarray,
                    R1: np.ndarray, t1: np.ndarray,
                    prefix: str = "") -> None:
    """Persist extrinsic transforms to ./camera_parameters/."""
    _ensure_dir("camera_parameters")

    with open(f"camera_parameters/{prefix}camera0_rot_trans.dat", "w") as fh:
        _write_matrix(fh, "R", R0)
        _write_matrix(fh, "T", t0)

    with open(f"camera_parameters/{prefix}camera1_rot_trans.dat", "w") as fh:
        _write_matrix(fh, "R", R1)
        _write_matrix(fh, "T", t1)


# -----------------------------------------------------------------------------#
# Calibration sanity-check
# -----------------------------------------------------------------------------#
def live_axis_overlay(
    cam0_key: str, cam1_key: str,
    K0: np.ndarray, d0: np.ndarray, R0: np.ndarray, t0: np.ndarray,
    K1: np.ndarray, d1: np.ndarray, R1: np.ndarray, t1: np.ndarray,
    z_shift: float = 50.0,
) -> None:
    """Draw projected XYZ axes on live feeds to visually verify calibration."""
    P0 = projection_matrix(K0, R0, t0)
    P1 = projection_matrix(K1, R1, t1)

    axes = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]], dtype=np.float32)
    axes = 5 * axes + np.array([0, 0, z_shift])

    pix0, pix1 = [], []
    for X in axes:
        Xh = np.append(X, 1)  # homogeneous
        pix0.append((P0 @ Xh)[:2] / (P0 @ Xh)[2])
        pix1.append((P1 @ Xh)[:2] / (P1 @ Xh)[2])
    pix0, pix1 = np.int32(pix0), np.int32(pix1)

    cap0 = cv.VideoCapture(calibration_settings[cam0_key])
    cap1 = cv.VideoCapture(calibration_settings[cam1_key])
    for cap in (cap0, cap1):
        cap.set(cv.CAP_PROP_FRAME_WIDTH, calibration_settings["frame_width"])
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, calibration_settings["frame_height"])

    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # X=R, Y=G, Z=B

    while True:
        ok0, f0 = cap0.read()
        ok1, f1 = cap1.read()
        if not (ok0 and ok1):
            sys.exit("[ERROR] Video stream interrupted")

        # Draw on camera0
        origin = tuple(pix0[0])
        for col, p in zip(colors, pix0[1:]):
            cv.line(f0, origin, tuple(p), col, 2)

        # Draw on camera1
        origin = tuple(pix1[0])
        for col, p in zip(colors, pix1[1:]):
            cv.line(f1, origin, tuple(p), col, 2)

        cv.imshow("Camera0", f0)
        cv.imshow("Camera1", f1)
        if cv.waitKey(1) & 0xFF == 27:
            break

    cv.destroyAllWindows()


# -----------------------------------------------------------------------------#
# Main driver
# -----------------------------------------------------------------------------#
def main() -> None:
    if len(sys.argv) != 2:
        sys.exit("Usage: python calibrate.py calibration_settings.yaml")

    load_settings(sys.argv[1])

    # Step 1 – capture mono frames
    capture_single_camera("camera0")
    capture_single_camera("camera1")

    # Step 2 – compute intrinsics
    K0, d0 = calibrate_intrinsics("frames/camera0*")
    save_intrinsics(K0, d0, "camera0")

    K1, d1 = calibrate_intrinsics("frames/camera1*")
    save_intrinsics(K1, d1, "camera1")

    # Step 3 – capture stereo pairs
    capture_stereo_pair("camera0", "camera1")

    # Step 4 – stereo calibration
    R01, t01 = stereo_calibrate(
        K0, d0, K1, d1,
        "frames_pair/camera0*", "frames_pair/camera1*"
    )

    # Step 5 – save extrinsics (camera0 is world origin)
    R0, t0 = np.eye(3, dtype=np.float32), np.zeros((3, 1), np.float32)
    save_extrinsics(R0, t0, R01, t01)

    # Optional – live check
    live_axis_overlay(
        "camera0", "camera1",
        K0, d0, R0, t0,
        K1, d1, R01, t01,
        z_shift=60.0
    )


if __name__ == "__main__":
    main()
