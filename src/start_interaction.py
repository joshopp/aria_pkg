import argparse
from aria_utils import AriaStreamer
import multiprocessing
import sys

from common import update_iptables, TerminalRawMode, exit_keypress
from gaze_processor import stream_image
from voice_controller import stream_audio
from feature_matching import match_features


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



# pipeline for complete interaction.
def main():
    args = parse_args()

    if args.update_iptables and sys.platform.startswith("linux"):
        update_iptables()

    # 1. create AriaStreamer instance and start streaming
    aria = AriaStreamer()
    device = aria.stream_start(args.device_ip, args.streaming_interface, args.profile_name)

    # 2. Use multiprocessing to run the img stream, audio stream, and feature matching pipelines
    # Press ESC or q to end loop
    while not exit_keypress():
        print("Starting iteration..")
        ctx = multiprocessing.get_context("spawn")
        img_proc = ctx.Process(target=stream_image)
        audio_proc = ctx.Process(target=stream_audio)
        matcher_proc = ctx.Process(target=match_features)

        img_proc.start()
        audio_proc.start()

        img_proc.join()
        print("Image streaming process finished.")

        matcher_proc.start()

        audio_proc.join()
        print("Audio streaming process finished.")

        matcher_proc.join()
        print("Feature matching process finished.")

    aria.stream_end(device)

if __name__ == "__main__":
    with TerminalRawMode():
        main()