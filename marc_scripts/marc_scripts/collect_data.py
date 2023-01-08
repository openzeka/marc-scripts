import rclpy
from rclpy.node import Node

from std_msgs.msg import String, Float64
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
import os
import numpy as np

class CollectData(Node):
        def __init__(self):
            super().__init__('collect_data')
            self.speed = None
            self.angle = None
            self.br = CvBridge()
            self.index = 0
            self.home_path = os.environ.get("HOME")
            self.training_data = np.array([0,0,0])
            datadir = self.home_path + '/marc_data/'
            
            if not os.path.exists(datadir):
                os.makedirs(datadir)
                
            i = 1
            while True:
                dname = datadir + '%03d'%i
                if os.path.exists(dname):
                    i += 1
                else:
                    os.makedirs(dname)
                    break

            self.path = dname+'/'
            self.file_name = self.path + '/training_data.npy'
            
            self.create_subscription(
                AckermannDriveStamped,
                '/ackermann_cmd',
                self.drive_call,
            10)
            
            self.create_subscription(
                Image,
                '/zed2/zed_node/rgb/image_rect_color',
                self.zed_callback,
            10)
        
        def drive_call(self, data):
            self.angle = data.drive.steering_angle
            self.speed = data.drive.speed
            
        def zed_callback(self, data):
            try:
                self.cv2_img = self.br.imgmsg_to_cv2(data)
                self.cv2_img = cv2.resize(self.cv2_img,(720,480),interpolation=cv2.INTER_AREA)
                try:
                
                    if self.cv2_img is None:
                        print('Camera could not detected!')
                    if self.speed is None:
                        print('Speed could not detected!')
                    if self.angle is None:
                        print('Angle could not detected!')

                    if not self.cv2_img is None and not self.speed is None and not self.angle is None:
                        fname = self.path+'%05d.jpg'%self.index
                        cv2.imshow('Image', self.cv2_img)
                        k = cv2.waitKey(10)
                        generated_data = np.array(['%05d.jpg'%self.index,self.speed,self.angle])
                        self.training_data = np.vstack((self.training_data, generated_data))

                        np.save(self.file_name, self.training_data)
                        cv2.imwrite(fname,self.cv2_img)	
                    self.index += 1
                    
                except Exception as e:
                    print('Hang on a sec...',e)
                    pass
                
            except CvBridgeError as e:
                print(e)
        
	
            
def main(args=None):
    rclpy.init(args=args)

    collect_data = CollectData()

    rclpy.spin(collect_data)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    collect_data.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()