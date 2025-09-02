import aria.sdk as aria
import csv
import cv2
import numpy as np
import os
import shutil
import threading
import time
import zmq

from aria_utils import AriaStreamer
from common import quit_keypress
import real_time_inference
from StreamingClientObserver import ImageObserver



def mk_cam_dir(save_path, id_map):
    #make camera header dirs
    for idx in id_map.keys():
        cam_folder = os.path.join(save_path, id_map[idx])
        print("new dir created: ", cam_folder)
        if os.path.exists(cam_folder):  
            shutil.rmtree(cam_folder)
        os.makedirs(cam_folder)


def mk_img_dir(save_path):
    #make dir where (undistorted) image .npy files are stored
    image_folder = os.path.join(save_path, 'images')
    print("new dir created: ", image_folder)
    if os.path.exists(image_folder):  
        shutil.rmtree(image_folder)
    os.makedirs(image_folder)

    undistorted_folder = os.path.join(save_path, 'undistorted_imgs')
    print("new dir created: ", undistorted_folder)
    if os.path.exists(undistorted_folder):  
        shutil.rmtree(undistorted_folder)
    os.makedirs(undistorted_folder)


def mk_et_dir_csv(eyetrack_folder):  
    # make eyetrack dir
    if os.path.exists(eyetrack_folder):
        shutil.rmtree(eyetrack_folder)
    os.makedirs(eyetrack_folder)
    print("new dir created: ", eyetrack_folder)
    # Initialize CSV file with headers
    eye_track_file_path = os.path.join(eyetrack_folder, "general_eye_gaze.csv")
    with open(eye_track_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow([
            "tracking_timestamp_ns", "yaw_rads_cpf", "pitch_rads_cpf", "depth_m_str",
            "yaw_low_rads_cpf", "pitch_low_rads_cpf", "yaw_high_rads_cpf", "pitch_high_rads_cpf", 
            "gaze_point_x", "gaze_point_y"
        ])


def save_eye_gaze_result_to_csv(result, file_path):
    # adds the new data to a csv
    with open(file_path, mode='a', newline='') as file:  # Open the file in append mode
        writer = csv.writer(file)
        writer.writerow(result) 


def control_command_listener():
    global saving_state
    saving_state = 0
    last_print_time = 0
    # connect to 0mq server to listen for "control" commands
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt_string(zmq.SUBSCRIBE, "command")  
    print("ZMQ socket connected to port 5556 for commands (Whisperer)")
    try:
        while True:
            _, command = socket.recv_string().split(" ", 1)
            current_time = time.time()
            if command == "WAIT":
                saving_state = 0
                if current_time - last_print_time >= 2: # print every two seconds
                    print("Subscriber is running, waiting for start command")
                    last_print_time = current_time
            elif command == "START" and saving_state == 0:
                saving_state = 1
                print("Start detected, saving data now")
            elif command == "END" and saving_state == 1:
                saving_state = 2
                print("Finish detected, stopping data saving")
                break
            else:
                print("Error: Invalid command received from speech recognition.")
    finally:
        socket.close()
            

def stream_image():
    filepath = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(filepath, "data")
    
    camera_id_map = {2: "rgbcam", 3: "eyetrack"}
    eyetrack_folder = os.path.join(save_path, "eyetracking")
    mk_cam_dir(save_path, camera_id_map)
    mk_et_dir_csv(eyetrack_folder)
    mk_img_dir(save_path)

    # 1. start interaction model to visualize et
    inference_model, device_calibration, rgb_label, rgb_camera_calibration, rgb_linear_camera_calibration = real_time_inference.eyetracking_initialization("cuda")

    # 2. subscribe to desired data channels
    img_streamer = AriaStreamer()
    data_channels = [aria.StreamingDataType.Rgb,
                     aria.StreamingDataType.EyeTrack]  
    message_size = 1  # adjust as needed, 1 is usually sufficient for real-time applications
    observer = img_streamer.stream_subscribe(data_channels, ImageObserver(rgb_camera_calibration, rgb_linear_camera_calibration, save_path, camera_id_map), message_size)
    
    # 3. listening thread for 0mq control msg
    threading.Thread(target=control_command_listener, daemon=True).start()

    # 4. start interaction model
    aria_window = "Meta Aria image"
    cv2.namedWindow(aria_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(aria_window, 1024, 1024)
    cv2.setWindowProperty(aria_window, cv2.WND_PROP_TOPMOST, 1)
    cv2.moveWindow(aria_window, 50, 50)   

    # 5. visualize data stream
    while not quit_keypress():
        value_mapping, eye_gaze_inference_result = real_time_inference.real_time_eyetracking(inference_model, observer.images, aria.CameraId.EyeTrack, observer.timestamp)
        if value_mapping is None:
            print("incorrect value mapping")
        gaze_point, image_with_gaze = real_time_inference.eye_tracking_visualization(device_calibration, rgb_camera_calibration, rgb_label, aria.CameraId.Rgb, observer.images, value_mapping) #, T_device_CPF)
        
        # # activate to permanently stream images
        # if image_with_gaze is not None:
        #     rotated_image = np.rot90(image_with_gaze, -1)
        #     color_img = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
        #     cv2.imshow(aria_window, color_img)
        # else:
        #     print("image with gaze is None")

        # visualize streaming and save when START command is detected
        if saving_state == 1:
            observer.save_flag = True
            if image_with_gaze is not None:
                image_with_gaze = np.array(image_with_gaze)
                rotated_image = np.rot90(image_with_gaze, -1)
                color_img = cv2.cvtColor(rotated_image, cv2.COLOR_RGB2BGR)
                cv2.imshow(aria_window, color_img)
                gaze_point_rotated = np.array([image_with_gaze.shape[0] - gaze_point[1] - 1, gaze_point[0]])
                eye_gaze_inference_result.extend(gaze_point_rotated.tolist()) # extend with coordinates of ET
                save_eye_gaze_result_to_csv(eye_gaze_inference_result, os.path.join(eyetrack_folder, "general_eye_gaze.csv"))
        if saving_state == 2:
            print("Stop listening to image data")
            break

    # 6. free resources
    img_streamer.streaming_client.unsubscribe()
    cv2.destroyAllWindows()
    print("Imaginary2 terminated, Image stream unsubscribed")   