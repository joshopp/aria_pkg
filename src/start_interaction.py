# Pipeline, die alle notwendigen Schritte für die Interaktion mit dem Roboter ausführt.
import argparse
import sys
from aria_utils import AriaStreamer
from common import update_iptables
from gaze_processor import stream_image
from voice_controller import stream_audio
# from feature_matching import match_features
import threading
import zmq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--update_iptables",
        default=False,
        action="store_true",
        help="Update iptables to enable receiving the data stream, only for Linux.",
    )
    parser.add_argument(
        "--interface",
        dest="streaming_interface",
        type=str,
        required=True,
        help="Type of interface to use for streaming. Options are usb or wifi.",
        choices=["usb", "wifi"],
    )
    parser.add_argument(
        "--profile",
        dest="profile_name",
        type=str,
        default="profile18",
        required=False,
        help="Profile to be used for streaming.",
    )
    parser.add_argument(
        "--device-ip", help="IP address to connect to the device over wifi"
    )
    return parser.parse_args()



def main():
    args = parse_args()

    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()
    
    context = zmq.Context()
    # 1. Create AriaStreamer instance and start streaming
    aria = AriaStreamer()
    device = aria.stream_start(args.device_ip, args.streaming_interface, args.profile_name)
    
    # 2. Start subscription and functions of image stream (imaginary2)
    img_threat = threading.Thread(target=stream_image, args=(context,), daemon=True) #TODO: daemon?
    audio_thread = threading.Thread(target=stream_audio, args=(context,), daemon=True) #TODO: daemon?
    
    img_threat.start()
    audio_thread.start()

    # matcher_threat = threading.Thread(target=match_features, args=(context,)) #TODO: daemon?
    # matcher_threat.start()

    img_threat.join()
    audio_thread.join()
    # matcher_threat.join()

    aria.stream_end(device)
    context.destroy()


    # threading.Thread(target=task_c).start()
if __name__ == "__main__":
    main()