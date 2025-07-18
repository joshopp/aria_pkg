Probleme mit CV2 and torch (-> SocialEye ET): pip uninstall av

AV and torchvision dont match



    A class for storing and manipulating inference results.

    This class provides comprehensive functionality for handling inference results from various
    Ultralytics models, including detection, segmentation, classification, and pose estimation.
    It supports visualization, data export, and various coordinate transformations.

    Attributes:
        orig_img (numpy.ndarray): The original image as a numpy array.
        orig_shape (Tuple[int, int]): Original image shape in (height, width) format.
        boxes (Boxes | None): Detected bounding boxes.
        masks (Masks | None): Segmentation masks.
        probs (Probs | None): Classification probabilities.
        keypoints (Keypoints | None): Detected keypoints.
        obb (OBB | None): Oriented bounding boxes.
        speed (dict): Dictionary containing inference speed information.
        names (dict): Dictionary mapping class indices to class names.
        path (str): Path to the input image file.
        save_dir (str | None): Directory to save results.

    Methods:
        update: Update the Results object with new detection data.
        cpu: Return a copy of the Results object with all tensors moved to CPU memory.
        numpy: Convert all tensors in the Results object to numpy arrays.
        cuda: Move all tensors in the Results object to GPU memory.
        to: Move all tensors to the specified device and dtype.
        new: Create a new Results object with the same image, path, names, and speed attributes.
        plot: Plot detection results on an input RGB image.
        show: Display the image with annotated inference results.
        save: Save annotated inference results image to file.
        verbose: Return a log string for each task in the results.
        save_txt: Save detection results to a text file.
        save_crop: Save cropped detection images to specified directory.
        summary: Convert inference results to a summarized dictionary.
        to_df: Convert detection results to a Pandas Dataframe.
        to_json: Convert detection results to JSON format.
        to_csv: Convert detection results to a CSV format.
        to_xml: Convert detection results to XML format.
        to_html: Convert detection results to HTML format.
        to_sql: Convert detection results to an SQL-compatible format.


        Panda PC: Cuda 12.2
        Driver 535.230.02
        torch 2.4.1
        pytorch3d 0.7.8

        Other PC: Cuda 12.8
        Driver 570.133.20
        torch 2.1.0 + cu118
        pytorch3d 0.7.8

        ros-wait_for-message: