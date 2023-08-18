import sys, getopt
import numpy as np
import cv2
import os
from gelsight import gsdevice
from gelsight import gs3drecon

def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)

def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  + 0.5

def main(argv):

    device = "mini"
    try:
       opts, args = getopt.getopt(argv, "hd:", ["device="])
    except getopt.GetoptError:
       print('python show3d.py -d <device>')
       sys.exit(2)
    for opt, arg in opts:
       if opt == '-h':
          print('show3d.py -d <device>')
          print('Use R1 for R1 device, and gsr15???.local for R2 device')
          sys.exit()
       elif opt in ("-d", "--device"):
          device = arg

    # Set flags 
    SAVE_VIDEO_FLAG = False
    GPU = False
    MASK_MARKERS_FLAG = True
    FIND_ROI = False

    # Path to 3d model
    path = '.'

    # Set the camera resolution
    # mmpp = 0.0887  # for 240x320 img size
    # mmpp = 0.1778  # for 160x120 img size from R1
    # mmpp = 0.0446  # for 640x480 img size R1
    # mmpp = 0.029 # for 1032x772 img size from R1
    mmpp = 0.075  # r2d2 gel 18x24mm at 240x320

    if device == "R1":
        finger = gsdevice.Finger.R1
        mmpp = 0.0887
    elif device[-5:] == "local":
        finger = gsdevice.Finger.R15
        mmpp = 0.0887
        capturestream = "http://" + device + ":8080/?action=stream"
    elif device == "mini":
        finger = gsdevice.Finger.MINI
        mmpp = 0.0625
    else:
        print('Unknown device name')
        print('Use R1 for R1 device \ngsr15???.local for R1.5 device \nmini for mini device')


    if finger == gsdevice.Finger.R1:
        dev = gsdevice.Camera(finger, 0)
        net_file_path = 'nnr1.pt'
    elif finger == gsdevice.Finger.R15:
        #cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
        dev = gsdevice.Camera(finger, capturestream)
        net_file_path = 'nnr15.pt'
    elif finger == gsdevice.Finger.MINI:
        # the device ID can change after unplugging and changing the usb ports.
        # on linux run, v4l2-ctl --list-devices, in the terminal to get the device ID for camera
        cam_id = gsdevice.get_camera_id("GelSight Mini")
        dev = gsdevice.Camera(finger, cam_id)
        net_file_path = 'nnmini.pt'

    dev.connect()

    ''' Load neural network '''
    model_file_path = path
    net_path = os.path.join(model_file_path, net_file_path)
    print('net path = ', net_path)

    if GPU: gpuorcpu = "cuda"
    else: gpuorcpu = "cpu"
    if device=="R1":
        nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R1, dev)
    else:
        nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15, dev)
    net = nn.load_nn(net_path, gpuorcpu)

    if SAVE_VIDEO_FLAG:
        #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
        file_path = './3dnnlive.mov'
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

    f0 = dev.get_raw_image()
    roi = (0, 0, f0.shape[1], f0.shape[0])

    if FIND_ROI:
        roi = cv2.selectROI(f0)
        roi_cropped = f0[int(roi[1]):int(roi[1] + roi[3]), int(roi[0]):int(roi[0] + roi[2])]
        cv2.imshow('ROI', roi_cropped)
        print('Press q in ROI image to continue')
        cv2.waitKey(0) 
        cv2.destroyAllWindows()
    elif f0.shape == (640,480,3) and device != 'mini':
        roi = (60, 100, 375, 380)
    elif f0.shape == (320,240,3) and device != 'mini':
        roi = (30, 50, 186, 190)
    elif f0.shape == (240,320,3) and device != 'mini':
        ''' cropping is hard coded in resize_crop_mini() function in gsdevice.py file '''
        border_size = 0 # default values set for mini to get 3d
        roi = (border_size,border_size,320-2*border_size,240-2*border_size) # default values set for mini to get 3d

    print('roi = ', roi)
    print('press q on image to exit')
    
    ''' use this to plot just the 3d '''
    if device == 'mini':
        vis3d = gs3drecon.Visualize3D(dev.imgh, dev.imgw, '', mmpp)
    else:
        vis3d = gs3drecon.Visualize3D(dev.imgw, dev.imgh, '', mmpp)

    try:
        while dev.while_condition:

            # get the roi image
            f1 = dev.get_image(roi)
            bigframe = cv2.resize(f1, (f1.shape[1]*2, f1.shape[0]*2))
            cv2.imshow('Image', bigframe)

            # compute the depth map
            dm = nn.get_depthmap(f1, MASK_MARKERS_FLAG)

            ''' Display the results '''
            vis3d.update(dm)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if SAVE_VIDEO_FLAG:
                out.write(f1)

    except KeyboardInterrupt:
            print('Interrupted!')
            dev.stop_video()

if __name__ == "__main__":
    main(sys.argv[1:])
