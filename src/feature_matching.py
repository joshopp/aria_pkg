from collections import Counter, defaultdict
import cv2
import json
import math
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import sys
import torch
from ultralytics import YOLO
import zmq

#import SuperGlue from local
superglue_path = "/home/jruopp/thesis_ws/src/superglue_pkg/SuperGluePretrainedNetwork/"
if superglue_path not in sys.path:
    sys.path.append(superglue_path)
from models.matching import Matching
from models.utils import (make_matching_plot, AverageTimer, read_image, frame2tensor)

# disable gradient calculation globally
torch.set_grad_enabled(False)



def superglue_matching_init():
    device = 'cuda' 
    config = {
        'superpoint': {
            'nms_radius': 2,
            'keypoint_threshold': 0.0,
            'max_keypoints': 10000,
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.00001,
        }
    }
    matching = Matching(config).eval().to(device)
    timer = AverageTimer(newline=True)
    return matching


def superglue(matching, aria_img, robo_img, bboxaria, viz_path):
    # compute tensor of images
    device = "cuda"
    aria_inp = frame2tensor(aria_img, device)
    robo_inp = frame2tensor(robo_img, device)

    # match both pictures and convert to numpy arrays
    pred = matching({'image0': aria_inp, 'image1': robo_inp})
    pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
    ariakpts, robokpts = pred['keypoints0'], pred['keypoints1']
    matches, conf = pred['matches0'], pred['matching_scores0']
    # print("Aria kpts:", len(ariakpts))
    # print("Robo kpts:", len(robokpts))
    print("Raw matches:", np.sum(matches > -1))

    # Write the matches to dictionary
    out_matches = {'keypoints0': ariakpts, 'keypoints1': robokpts,
                    'matches': matches, 'match_confidence': conf}
    # filter matches by bbox
    ariakpts, matches, conf, valid_bbox = filter_points_by_bbox(ariakpts, matches, conf, bboxaria)

    # keep valid matched keypoints.
    valid = matches > -1
    aria_matched = ariakpts[valid]
    robo_matched = robokpts[matches[valid]]
    mconf = conf[valid & valid_bbox] #conf values of valid matches in bbox

    # Visualize the matches
    color = cm.jet(mconf)
    text = []
    make_matching_plot(
        aria_img, robo_img, ariakpts, robokpts,
        aria_matched, robo_matched,
        color, text, viz_path)
    return aria_matched, robo_matched, mconf


def filter_points_by_bbox(keypoints, matches, conf, bbox):
    x_min, y_min, x_max, y_max = bbox
    tolerance = 30
    x_min -= tolerance
    y_min -= tolerance
    x_max += tolerance
    y_max += tolerance

    valid = (keypoints[:, 0] >= x_min) & (keypoints[:, 0] <= x_max) & \
            (keypoints[:, 1] >= y_min) & (keypoints[:, 1] <= y_max)
    print(f"Valid keypoints in bbox: {np.sum(valid)}")
    print(f"Valid matches in bbox: {np.sum(matches[valid] > -1)}")
    return keypoints[valid], matches[valid], conf[valid], valid[valid]


def calculate_matching_points_in_box(mkpts1, boxes):
    points_per_bbox = []
    tolerance = 15 
    for box in boxes:
        bbox_tensor = box.xyxy[0]  # Bbbox coordinates as tensor
        x_min, y_min, x_max, y_max = bbox_tensor.tolist()
        # check which mkpt is in box
        valid = (mkpts1[:, 0] >= x_min-tolerance) & (mkpts1[:, 0] <= x_max+tolerance) & \
                (mkpts1[:, 1] >= y_min-tolerance) & (mkpts1[:, 1] <= y_max+tolerance)
        # calculate sum
        points_per_bbox.append(np.sum(valid))

    max_bbox_index = np.argmax(points_per_bbox)
    return max_bbox_index, points_per_bbox


def get_robo_img_bbox(context, maskModel):
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.159.6.33:5560") #IP of Panda3 PC
    print("ZMQ socket connected to Panda3 PC")

    # request image from IntelRealSense camera (roboter)
    socket.send(b"send_image")
    print("Request sent to Panda3 PC for image...")

    # receive image in bytes, decode to NumPy array and convert to grayscale
    img_bytes = socket.recv()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)  # BGR format
    # img = cv2.imread('/home/jruopp/thesis_ws/src/aria_pkg/data/superglue/sent_img2.png')
    if img is not None:
        print(f"Received image from robot")

    rot_img = np.rot90(img, -2)
    gray_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2GRAY)

    # get bounding boxes from YOLO
    registered_bricks = get_brick_poses(rot_img, maskModel)
    annotated_frame = registered_bricks.plot()  

    # visualize images if needed
    # cv2.imshow("Robo Image", rot_img)
    # cv2.imshow("Gray Image", gray_img)
    # cv2.imshow("BBoxes", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    bboxesrobo = []
    for brick in registered_bricks.boxes:
        bbox_tensor = brick.xyxy[0]  # Bbbox coordinates as tensor
        x1, y1, x2, y2 = bbox_tensor.tolist()
        bboxesrobo.append([x1, y1, x2, y2])
    socket.close()

    return gray_img, registered_bricks.boxes


def get_aria_img_bbox(context, maskModel, et_csv_file_path, rgb_csv_file_path):
    socket = context.socket(zmq.PULL)
    socket.bind("tcp://*:5557")
    print("ZMQ socket bound to port 5557")

    # receive msg from whisperer
    print("Waiting for message from Whisperer...\n")
    gpt_string = socket.recv_string()
    gpt_json = json.loads(gpt_string)
    if gpt_string is None:
        print("No response received, return None")
        return None
    else:
        print(f"Received String object as response: {gpt_string}")
    
    # extract timestamps and calculate gaze_points from pitch and yaw
    first_timestamp = gpt_json.get('startTime_ns', [])
    last_timestamp = gpt_json.get('endTime_ns', [])
    # first_timestamp = 4276327516875
    # last_timestamp = 4276787516875
    et_data = extract_yaw_pitch(et_csv_file_path, first_timestamp, last_timestamp )
    filename = extract_timestamp(rgb_csv_file_path, first_timestamp)

    gaze_points = et_data[['gaze_point_x', 'gaze_point_y']].to_numpy()
    # print(f"filename {filename}, gaze_points: {gaze_points}")
    # open RGB img from first timestamp and convert to grayscale
    path = "/home/jruopp/thesis_ws/src/aria_pkg/data/undistorted_imgs/"
    file = path+filename

    img = np.load(file)
    rot_img = np.rot90(img, -1)
    color_img = cv2.cvtColor(rot_img, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    registered_bricks = get_brick_poses(color_img, maskModel)
    annotated_frame = registered_bricks.plot()  

    # # visualize images if needed
    # cv2.imshow("Aria Image", color_img)
    # cv2.imshow("Gray Image", gray_img)
    # cv2.imshow("BBoxes", annotated_frame)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    # get bounding boxes from YOLO
    
    if registered_bricks is None:
        print("No bricks detected, returning original image")
        return gray_img, None
    foc_box = find_most_focused_bbox(gaze_points, registered_bricks.boxes)
    print(f"Most focused bbox: {foc_box}")

    # socket.close()
    return gray_img, foc_box['coordinates']


def extract_timestamp(csv_file, first_timestamp):
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="#timestamp [ns]")

    start_row = df.iloc[(df['#timestamp [ns]'] - first_timestamp).abs().idxmin()]
    start_idx = start_row.name
    filename = df.loc[start_idx, 'filename']
    return filename


def extract_yaw_pitch(csv_file, first_timestamp, last_timestamp):
    # Load CSV into a DataFrame and sort by timestamps
    df = pd.read_csv(csv_file)
    df = df.sort_values(by="tracking_timestamp_ns")
    
    # Find the closest rows to first_timestamp and last_timestamp
    start_row = df.iloc[(df['tracking_timestamp_ns'] - first_timestamp).abs().idxmin()]
    end_row = df.iloc[(df['tracking_timestamp_ns'] - last_timestamp).abs().idxmin()]

    # Get indices for slice data inbetween timestamps
    start_idx = start_row.name
    end_idx = end_row.name
    print(f"slice data between timestamps {start_idx} and {end_idx}")
    if start_idx > end_idx:
        start_idx, end_idx = end_idx, start_idx  # Ensure start_idx is smaller

    # get data from relevant rows
    result = df.loc[start_idx:end_idx, ['tracking_timestamp_ns', 'gaze_point_x', 'gaze_point_y']]
    return result



def get_brick_poses(color_image, maskModel):
    registered_bricks = maskModel(
        color_image, iou=0.9, verbose=False)[0]
    if not registered_bricks or registered_bricks[0] is None:
        print("Warnung: Keine Erkennung")
        return color_image  # Rückgabe des Originals als Fallback
    return registered_bricks


def find_most_focused_bbox(gaze_points, result_boxes):
    focus_counter = Counter()
    bbox_lookup = {}
    distances_per_bbox = defaultdict(list)

    for gaze_point in gaze_points:
        focused = find_focused_bbox(gaze_point, result_boxes)
        # print(f"Focused bbox: {focused}")
        if focused:
            key = focused["bbox_index"]
            focus_counter[key] += 1
            distances_per_bbox[key].append(focused["distance"])
            if key not in bbox_lookup:
                bbox_lookup[key] = focused

    if not focus_counter:
        return None

    most_common_key, count = focus_counter.most_common(1)[0]
    avg_distance = sum(distances_per_bbox[most_common_key]) / len(distances_per_bbox[most_common_key])

    result = bbox_lookup[most_common_key].copy()
    result["focus_count"] = count
    result["distance"] = avg_distance
    return result


def find_focused_bbox(gaze_point, result_boxes):
    """
    returns closest BBox to gaze_point
    result_boxes: Ultralytics Result.boxes.xyxy
    """
    x, y = gaze_point
    closest_bbox = None
    min_dist = float('inf')
    i=0
    
    for brick in result_boxes:
        bbox_tensor = brick.xyxy[0]  # Bbbox coordinates as tensor
        x1, y1, x2, y2 = bbox_tensor.tolist()

        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        distance = math.hypot(center_x - x, center_y - y)
        # print(f"Distance to bbox {i}: {distance}")
        i += 1
        if distance < min_dist and distance < 150:  # filter out if no box is focused
            min_dist = distance
            closest_bbox = {
                "confidence": brick.conf[0],
                "class": brick.cls[0],
                "bbox_index": i,
                "coordinates": (x1, y1, x2, y2),
                "center": (center_x, center_y),
                "distance": distance
            }
    return closest_bbox     



def publish_bbox(context, bbox, image_shape):
    # extract most important values from bbox into dictionary
    bbox_tensor = bbox.xyxy[0]  # Bbbox coordinates as tensor
    x1, y1, x2, y2 = bbox_tensor.tolist()
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    height, width = image_shape[:2]
    flipped_center = (width - center_x, height - center_y)
    bbox_dict = {
        "confidence": bbox.conf[0],
        "class": bbox.cls[0],
        "coordinates": (x1, y1, x2, y2),
        "center": flipped_center,
        }

    socket = context.socket(zmq.REQ)
    socket.connect("tcp://10.159.6.33:5559") #IP of Panda3 PC
    print("ZMQ socket connected to Panda3 PC")
    
    # send bbox center to Panda3 PC
    socket.send_json(bbox_dict['center'])
    print("Bounding Box to grab sent to Panda3 PC...")

    # receive answer, whether correct brick was grabbed
    success = socket.recv_json()
    if success:
        print("Brick successfully grabbed by Panda")
    else:
        print("Error: Brick not grabbed by Panda")
    socket.close()


def match_features():
    maskModel = YOLO('/home/jruopp/thesis_ws/src/aria_pkg//src/best.pt')
    et_csv_file_path = '/home/jruopp/thesis_ws/src/aria_pkg/data/eyetracking/general_eye_gaze.csv' 
    rgb_csv_file_path = '/home/jruopp/thesis_ws/src/aria_pkg/data/rgbcam/data.csv' 
    viz_path = '/home/jruopp/thesis_ws/src/aria_pkg/data/superglue/matchresult.png'
    context = zmq.Context()

    # 1. Initialize SuperGlue
    print("Initializing SuperGlue")
    matching = superglue_matching_init()
    
    # 2. get images and bounding boxes from Aria and Robo
    robo_img, bboxesrobo  = get_robo_img_bbox(context, maskModel)
    print(f"Panda Intel: Image and coordinates of Bounding Boxes found") #: {bboxesrobo}")
    aria_img, bboxaria = get_aria_img_bbox(context, maskModel, et_csv_file_path, rgb_csv_file_path)
    print(f"Meta Aria glasses: Image and coordinates of focussed Bounding Box found") #: {bboxaria}")

    # 3. Start matching features and compute bbox with gaze
    print("Start matching features...")
    aria_matched, robo_matched, mconf = superglue(matching, aria_img, robo_img, bboxaria, viz_path)
    print(f"matched Aria points") # : {aria_matched}")
    matched_bbox, points_per_bbox = calculate_matching_points_in_box(robo_matched, bboxesrobo)
    targeted_bbox = bboxesrobo[matched_bbox]
    print(f"Matched bounding box index: {matched_bbox} with {points_per_bbox} points.")

    # 4. publish the Bbox to Panda3 PC for grasping
    publish_bbox(context, targeted_bbox, robo_img.shape)
    # example: Coordinates: [496.7156066894531, 66.25860595703125, 550.5313110351562, 96.8639144897461]


#context = zmq.Context()
#match_features(context)

# {
#   "words": [
#     "this"
#   ],
#   "object_name": [
#     "brick"
#   ],
#   "startTime_ns": [
#     2946642201652
#   ],
#   "endTime_ns": [
#     2947122201652
#   ],
#   "question": [
#     "please grab this yellow brick"
#   ]
# }



