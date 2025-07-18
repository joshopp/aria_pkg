from foundation_pose.PoseEstimationApp import PoseEstimatorApp
from foundation_pose.real_sense_reader import RealSenseReader
from ultralytics import YOLO


def grab_brick(self, brick):
    # TODO: get brick via zmq. brick: [center_pose, size, color, mask, brick_class_id]

    webcam = RealSenseReader()
    maskModel = YOLO('/home/panda3/Desktop/Robot_BA/best.pt')
    robot = PoseEstimatorApp(reader=webcam, maskModel=maskModel)
    
    
    grip, free_brick = robot.get_collision_free_bricks_and_grips(brick)
    self.robot.sort_brick(grip, True)