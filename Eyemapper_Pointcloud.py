#Pointcloud Depth Map Generation - Acquiring real-life coordinates using stereo cameras.
#Young Kim - 2022

import time
import cv2
import numpy as np
import json
from datetime import datetime
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar blue
property uchar green
property uchar red
end_header
'''


print ("You can press 'Q' to quit this script!")
time.sleep (5)

# Visualization settings
showDisparity = True
showUndistortedImages = True

# Depth map default preset
SWS = 5
PFS = 5
PFC = 29
MDS = -30
NOD = 160
TTH = 100
UR = 10
SR = 14
SPWS = 100

# Camera settimgs
img_width = 3840
img_height = 1088

# Image for disparity
imageToDisp = './scenes4test/scene_3840x1088_20.png'


# Depth Map colors autotune
autotune_min = 10000000
autotune_max = -10000000


# Initialize interface windows
cv2.namedWindow("Image")
cv2.moveWindow("Image", 50,100)
cv2.namedWindow("left")
cv2.moveWindow("left", 450,100)
cv2.namedWindow("right")
cv2.moveWindow("right", 850,100)


disparity = np.zeros((img_width, img_height), np.uint8)
sbm = cv2.StereoBM_create(numDisparities=0, blockSize=21)

def stereo_depth_map(rectified_pair):
    dmLeft = rectified_pair[0]
    dmRight = rectified_pair[1]
    disparity = sbm.compute(dmLeft, dmRight)
    local_max = disparity.max()
    local_min = disparity.min()
    #print(local_max, local_min)
    # "Jumping colors" protection for depth map visualization
    global autotune_max, autotune_min
    autotune_max = max(autotune_max, disparity.max())
    autotune_min = min(autotune_min, disparity.min())

    disparity_grayscale = (disparity-autotune_min)*(65535.0/(autotune_max-autotune_min))
    disparity_fixtype = cv2.convertScaleAbs(disparity_grayscale, alpha=(255.0/65535.0))
    disparity_color = cv2.applyColorMap(disparity_fixtype, cv2.COLORMAP_JET)
    return disparity_color, disparity_fixtype, disparity.astype(np.float32) / 16.0

def load_map_settings( fName ):
    global SWS, PFS, PFC, MDS, NOD, TTH, UR, SR, SPWS, loading_settings
    print('Loading parameters from file...')
    f=open(fName, 'r')
    data = json.load(f)
    SWS=data['SADWindowSize']
    PFS=data['preFilterSize']
    PFC=data['preFilterCap']
    MDS=data['minDisparity']
    NOD=data['numberOfDisparities']
    TTH=data['textureThreshold']
    UR=data['uniquenessRatio']
    SR=data['speckleRange']
    SPWS=data['speckleWindowSize']    
    #sbm.setSADWindowSize(SWS)
    sbm.setPreFilterType(1)
    sbm.setPreFilterSize(PFS)
    sbm.setPreFilterCap(PFC)
    sbm.setMinDisparity(MDS)
    sbm.setNumDisparities(NOD)
    sbm.setTextureThreshold(TTH)
    sbm.setUniquenessRatio(UR)
    sbm.setSpeckleRange(SR)
    sbm.setSpeckleWindowSize(SPWS)
    f.close()
    print ('Depth map settings are loaded from the file '+fName)

# Loading depth map settings
load_map_settings ("3dmap_set.txt")

# Loading stereoscopic calibration data
try:
    npzfile = np.load('./calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
except:
    print("Camera calibration data not found in cache, file ", './calibration_data/{}p/stereo_camera_calibration.npz'.format(img_height))
    exit(0)
try:
    npzright = np.load('./calibration_data/{}p/camera_calibration_right.npz'.format(img_height))
except:
    print("Camera calibration data not found in cache, file ", './calibration_data/{}p/camera_calibration_right.npz'.format(img_height))
    exit(0)    
    
    
imageSize = tuple(npzfile['imageSize'])
leftMapX = npzfile['leftMapX']
leftMapY = npzfile['leftMapY']
rightMapX = npzfile['rightMapX']
rightMapY = npzfile['rightMapY']
QQ = npzfile['dispartityToDepthMap']
right_K = npzright['camera_matrix']
#print (right_K)
#print (QQ)
#exit(0)

map_width = 1920
map_height = 1088

min_y = 10000
max_y = -10000
min_x =  10000
max_x = -10000
min_z =  10000
max_z = -10000

def remove_invalid(disp_arr, points, colors ):
    mask = (
        (disp_arr > disp_arr.min()) &
        #(disp_arr < disp_arr.max()) &
        np.all(~np.isnan(points), axis=1) &
        np.all(~np.isinf(points), axis=1) 
    )    
    return points[mask], colors[mask]

def calc_point_cloud(image, disp, q):
    points = cv2.reprojectImageTo3D(disp, q).reshape(-1, 3)
    minz = np.amin(points[:,2])
    maxz = np.amax(points[:,2])
    print("Min Z: " + str(minz))
    print("Max Z: " + str(maxz))
    
    #if our image is color or black and white?
    image_dim = image.ndim
    if (image_dim == 2):  # grayscale
        colors = image.reshape(-1, 1)
    elif (image_dim == 3): #color
        colors = image.reshape(-1, 3)
    else:
        print ("Wrong image data")
        exit (0)
    return remove_invalid(disp.reshape(-1), points, colors)

def calc_projected_image(points, colors, r, t, k, dist_coeff, width, height):
    xy, cm = project_points(points, colors, r, t, k, dist_coeff, width, height)
    image = np.zeros((height, width, 3), dtype=colors.dtype)
    image[xy[:, 1], xy[:, 0]] = cm
    return image

def project_points(points, colors, r, t, k, dist_coeff, width, height):
    projected, _ = cv2.projectPoints(points, r, t, k, dist_coeff)
    xy = projected.reshape(-1, 2).astype(np.int)
    mask = (
        (0 <= xy[:, 0]) & (xy[:, 0] < width) &
        (0 <= xy[:, 1]) & (xy[:, 1] < height)
    )
    colorsreturn = colors[mask]
    return xy[mask], colorsreturn

def rotate(arr, anglex, anglez):
    return np.array([  # rx
        [1, 0, 0],
        [0, np.cos(anglex), -np.sin(anglex)],
        [0, np.sin(anglex), np.cos(anglex)]
    ]).dot(np.array([  # rz
        [np.cos(anglez), 0, np.sin(anglez)],
        [0, 1, 0],
        [-np.sin(anglez), 0, np.cos(anglez)]
    ])).dot(arr)

angles = {  # x, z
    'w': (-np.pi/40, 0),
    's': (np.pi/40, 0),
    'a': (0, np.pi/40),
    'd': (0, -np.pi/40)
    }
r = np.eye(3)
t = np.array([0, 0.0, 100.5])

def write_ply(fn, verts, colors):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'wb') as f:
        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')
    print ("Pointcloud saved to "+fn)


while 1:
    pair_img = cv2.imread(imageToDisp,0)
    
    # Cutting stereopair to the left and right images
    imgLeft = pair_img [0:img_height,0:int(img_width/2)] #Y+H and X+W
    imgRight = pair_img [0:img_height,int(img_width/2):img_width] #Y+H and X+W
    
    # Undistorting images
    imgL = cv2.remap(imgLeft, leftMapX, leftMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    imgR = cv2.remap(imgRight, rightMapX, rightMapY, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rectified_pair = (imgL, imgR)
    imgLtoShow = cv2.resize (imgL, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
    imgRtoShow = cv2.resize (imgR, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
        
    # Disparity map calculation
    disparity, disparity_bw, native_disparity  = stereo_depth_map(rectified_pair)
    disparity_to_show = cv2.resize (disparity, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)

    # Point cloud calculation   
    points_3, colors = calc_point_cloud(disparity, native_disparity, QQ)

    # Camera settings for the pointcloud projection 
    k = right_K
    dist_coeff = np.zeros((4, 1))
    
    if (showUndistortedImages):
        cv2.imshow("left", imgLtoShow)
        cv2.imshow("right", imgRtoShow)    
    t2 = datetime.now()
    if (showDisparity):
        cv2.imshow("Image", disparity_to_show)
    key = cv2.waitKey(1) & 0xFF   
    if key == ord("q"):
        write_ply("output.ply", points_3, colors)
        quit();
    ch = chr(key)
    if ch in angles:
        ax, az = angles[ch]
        r = rotate(r, -ax, -az)
        #t = rotate(t, ax, az)
    if ch == "1":   # decrease camera distance from the point cloud
        t[2] -= 100
    elif ch == "2": # decrease camera distance from the point cloud
        t[2] += 100
    
    projected_image = calc_projected_image(points_3, colors, r, t, k, dist_coeff, map_width, map_height)
    projected_image_toshow = cv2.resize (projected_image, dsize=(640, 362), interpolation = cv2.INTER_CUBIC)
    cv2.imshow("XY projection", projected_image_toshow)
