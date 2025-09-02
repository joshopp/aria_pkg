import aria.sdk as aria
import csv
import numpy as np
import os
from projectaria_tools.core import calibration
from projectaria_tools.core.sensor_data import ImageDataRecord
from projectaria_tools.core.sensor_data import AudioDataRecord, AudioData
from scipy.signal import resample



class ImageObserver:
    def __init__(self, rgb_camera_calibration, rgb_linear_camera_calibration, save_path, camera_id_map):
        self.images = {}
        self.timestamp = 0
        self.save_flag = False
        self.rgb_camera_calibration = rgb_camera_calibration
        self.rgb_linear_camera_calibration = rgb_linear_camera_calibration
        self.save_path = save_path
        self.camera_id_map = camera_id_map


    def on_image_received(self, image: np.array, record: ImageDataRecord):
        self.images[record.camera_id] = image
        self.timestamp = record.capture_timestamp_ns
        timestamp_ns = self.timestamp
        save_folder = os.path.join(self.save_path, self.camera_id_map[record.camera_id])
        
        # save image if save_flag is True
        if self.save_flag:
            if not os.path.exists(save_folder):  
                print(f"ERROR: Folder {save_folder} does not exist, create folder")
                os.makedirs(save_folder)
            csv_path = os.path.join(save_folder,"data.csv")
            csv_header = ["#timestamp [ns]", "filename"]

            # save csv file with timestamp and corresponding img for all cam_ids
            if not os.path.exists(csv_path):
                with open(csv_path, mode="w", newline="") as csv_file:
                    writer = csv.writer(csv_file)
                    writer.writerow(csv_header) 
            with open(csv_path, mode="a", newline="") as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([timestamp_ns, f"{timestamp_ns}.npy"])

            # save all images
            image_path = os.path.join(self.save_path, 'images')
            np.save(os.path.join(image_path, f"{timestamp_ns}.npy"), image)

            # undistort RGB image
            if record.camera_id == aria.CameraId.Rgb:
                # undistort img
                undistort_image = calibration.distort_by_calibration(
                    image,
                    self.rgb_linear_camera_calibration,
                    self.rgb_camera_calibration,
                )
                # save undistorted image
                undistort_path = os.path.join(self.save_path, "undistorted_imgs")
                np.save(os.path.join(undistort_path, f"{timestamp_ns}.npy"), undistort_image)


class AudioObserver:
    def __init__(self):
        self.whisper_rate = 16000 # sample rate faster-whisper = 16000
        self.aria_rate= 48000 # sample rate Aria = 48000
        self.audio = []
        self.audios = [[] for c in range(7)]
        self.sampled_audios = np.zeros(self.aria_rate * 1, dtype=np.int8)
        self.timestamp = []
        self.timestamps = []
        self.received = False
        self.last_len = 0

    # source sample rate to 16k
    def resample_audio(self):
        starttime_ns = np.copy(self.timestamps[0])
        audios = np.copy(np.array(self.audios))
        num_samples = int(len(audios[0]) * self.whisper_rate / self.aria_rate)
        sampled_audios = resample(np.mean(np.array(audios), axis=0), num_samples)
        sampled_audios = sampled_audios / 1e8 # normalize sound intensity
        sampled_audios = sampled_audios.astype(np.float32)
        return sampled_audios, starttime_ns
    
    # source sample rate to 16k for saving audios as wav
    def resample_audio_wav(self):
        audios = np.copy(np.array(self.audios))
        current_len = len(audios[1])
        if current_len <= self.last_len:
            return None, None
        # only save new part
        new_audios = [ch[self.last_len:] for ch in audios]
        self.last_len = current_len
        mixed = np.mean(np.array(new_audios), axis=0)

        # Resample von 48k -> 16k
        num_samples = int(len(new_audios[0]) * self.whisper_rate / self.aria_rate)
        sampled_audios = resample(mixed, num_samples)

        # normalize to [-1,1\
        max_val = np.max(np.abs(sampled_audios))
        if max_val > 0:
            sampled_audios = sampled_audios / max_val
        return sampled_audios.astype(np.float32), None
    

    def on_audio_received(self, audio_data: AudioData, record: AudioDataRecord):
        self.audio, self.timestamp = audio_data.data, record.capture_timestamps_ns          
        self.timestamps += record.capture_timestamps_ns   
        
        # Record Limitation: 100s;10 samples per second
        rec_limit = self.aria_rate* 10 * 100       
        if len(self.timestamps) >= rec_limit:
            del self.timestamps[-rec_limit]

        # save data to audios
        for c in range(7):
            self.audios[c] += self.audio[c::7]
            if len(self.audios[c]) >= rec_limit:
                del self.audios[c][-rec_limit:]

        self.received = True


