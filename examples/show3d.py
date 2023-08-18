import sys, getopt
import numpy as np
import cv2
import os
from gelsight import gsdevice
from gelsight import gs3drecon

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

# Set flags 
SAVE_VIDEO_FLAG = False
GPU = False
MASK_MARKERS_FLAG = True
FIND_ROI = False

def get_diff_img(img1, img2):
    return np.clip((img1.astype(int) - img2.astype(int)), 0, 255).astype(np.uint8)

def get_diff_img_2(img1, img2):
    return (img1 * 1.0 - img2) / 255.  + 0.5

class Show3D(Node):

    def __init__(self, argv):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(Float64MultiArray, '/gelsight_touch', 10)
        timer_period = 0.04  # seconds
        self.dev = None
        self.device = "mini"
        self.init_dev(argv)
        self.timer = self.create_timer(timer_period, self.timer_callback)
        

    
    def init_dev(self,argv):
        

        try:
            opts, args = getopt.getopt(argv, "hd:", ["self.device="])
        except getopt.GetoptError:
            print('python show3d.py -d <self.device>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('show3d.py -d <self.device>')
                print('Use R1 for R1 self.device, and gsr15???.local for R2 self.device')
                sys.exit()
            elif opt in ("-d", "--self.device"):
                self.device = arg

       
        # Path to 3d model
        path = '.'

        # Set the camera resolution
        # mmpp = 0.0887  # for 240x320 img size
        # mmpp = 0.1778  # for 160x120 img size from R1
        # mmpp = 0.0446  # for 640x480 img size R1
        # mmpp = 0.029 # for 1032x772 img size from R1
        mmpp = 0.075  # r2d2 gel 18x24mm at 240x320

        if self.device == "R1":
            finger = gsdevice.Finger.R1
            mmpp = 0.0887
        elif self.device[-5:] == "local":
            finger = gsdevice.Finger.R15
            mmpp = 0.0887
            capturestream = "http://" + self.device + ":8080/?action=stream"
        elif self.device == "mini":
            finger = gsdevice.Finger.MINI
            mmpp = 0.0625
        else:
            print('Unknown self.device name')
            print('Use R1 for R1 self.device \ngsr15???.local for R1.5 self.device \nmini for mini self.device')


        if finger == gsdevice.Finger.R1:
            self.dev = gsdevice.Camera(finger, 0)
            net_file_path = 'nnr1.pt'
        elif finger == gsdevice.Finger.R15:
            #cap = cv2.VideoCapture('http://gsr15demo.local:8080/?action=stream')
            self.dev = gsdevice.Camera(finger, capturestream)
            net_file_path = 'nnr15.pt'
        elif finger == gsdevice.Finger.MINI:
            # the self.device ID can change after unplugging and changing the usb ports.
            # on linux run, v4l2-ctl --list-devices, in the terminal to get the self.device ID for camera
            cam_id = gsdevice.get_camera_id("GelSight Mini")
            self.dev = gsdevice.Camera(finger, cam_id)
            net_file_path = 'nnmini.pt'

        self.dev.connect()

        ''' Load neural network '''
        model_file_path = path
        net_path = os.path.join(model_file_path, net_file_path)
        print('net path = ', net_path)

        if GPU: gpuorcpu = "cuda"
        else: gpuorcpu = "cpu"
        if self.device=="R1":
            self.nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R1, self.dev)
        else:
            self.nn = gs3drecon.Reconstruction3D(gs3drecon.Finger.R15, self.dev)
        self.net = self.nn.load_nn(net_path, gpuorcpu)

        if SAVE_VIDEO_FLAG:
            #### Below VideoWriter object will create a frame of above defined The output is stored in 'filename.avi' file.
            file_path = './3dnnlive.mov'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter(file_path, fourcc, 60, (160, 120), isColor=True)

        f0 = self.dev.get_raw_image()
        self.roi = (0, 0, f0.shape[1], f0.shape[0])
        # self.roi = (20, 10, 280, 220)

        if FIND_ROI:
            self.roi = cv2.selectROI(f0)
            roi_cropped = f0[int(self.roi[1]):int(self.roi[1] + self.roi[3]), int(self.roi[0]):int(self.roi[0] + self.roi[2])]
            cv2.imshow('self.roi', roi_cropped)
            print('Press q in self.roi image to continue')
            cv2.waitKey(0) 
            cv2.destroyAllWindows()
        elif f0.shape == (640,480,3) and self.device != 'mini':
            self.roi = (60, 100, 375, 380)
        elif f0.shape == (320,240,3) and self.device != 'mini':
            print("I'm here")
            self.roi = (30, 50, 186, 190)
        elif f0.shape == (240,320,3) and self.device != 'mini':
            ''' cropping is hard coded in resize_crop_mini() function in gsdevice.py file '''
            border_size = 0 # default values set for mini to get 3d
            self.roi = (border_size,border_size,320-2*border_size,240-2*border_size) # default values set for mini to get 3d

        print('roi = ', self.roi)
        print('press q on image to exit')
        
        ''' use this to plot just the 3d '''
        if self.device == 'mini':
            self.vis3d = gs3drecon.Visualize3D(self.dev.imgh, self.dev.imgw, '', mmpp)
        else:
            self.vis3d = gs3drecon.Visualize3D(self.dev.imgw, self.dev.imgh, '', mmpp)

    def timer_callback(self):
        self.print_once = False
        try:
            depths = []
            while self.dev.while_condition:

                # get the roi image
                f1 = self.dev.get_image(self.roi)
                bigframe = cv2.resize(f1, (f1.shape[1]*2, f1.shape[0]*2))
                cv2.imshow('Image', bigframe)

                # compute the depth map
                dm = self.nn.get_depthmap(f1, MASK_MARKERS_FLAG)

                ''' Display the results '''
                self.vis3d.update(dm)
                
                points3d = self.vis3d.get_depth_points()


                depths.append(points3d[:,2])
                if len(depths) > 20:
                    depths.pop(0)
                
                depth_average = np.average(np.matrix(depths), axis=0)
                
                #print(depth_average.shape)
                #depth_average = np.average(depths, axis=0)
                all_max_index = np.where(depth_average > 0.6)
                # print('depth is = ', [x for x in depths if x > 1.0])

                if len(all_max_index[0]) == 0:
                    # corinate_x_average = self.dev.imgw/2
                    # corinate_y_average = self.dev.imgh/2
                    corinate = np.array([self.dev.imgw/2, self.dev.imgh/2, 0])
                    # print('len(all_max_index[0]) =',len(all_max_index))
                #     average_depth = 0
                else:
                    # corinates = points3d[all_max_index]
                #     corinate_x_min = np.min(corinates[:,0])
                #     corinate_x_max = np.max(corinates[:,0])
                #     corinate_y_min = np.min(corinates[:,1])
                #     corinate_y_max = np.max(corinates[:,1])

                #     average_depth = np.average(depths[all_max_index])

                #     corinate_x_average = (corinate_x_min + corinate_x_max)/2
                #     corinate_y_average = (corinate_y_min + corinate_y_max)/2
                    max_depth_index = np.argmax(depth_average)
                    # if not self.print_once:
                        # self.print_once = True
                    corinate = points3d[max_depth_index,:]


                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                if SAVE_VIDEO_FLAG:
                    self.out.write(f1)
                
                msg = Float64MultiArray()
                msg.data.append(corinate[0])
                msg.data.append(corinate[1])
                msg.data.append(corinate[2])
                msg.data.append(self.dev.imgw)
                msg.data.append(self.dev.imgh)
                self.publisher_.publish(msg)

        except KeyboardInterrupt:
            print('Interrupted!')
            self.dev.stop_video()
    


def main(argv):
    rclpy.init(args=argv)
    show3d = Show3D(argv)
    rclpy.spin(show3d)

    show3d.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main(sys.argv[1:])
