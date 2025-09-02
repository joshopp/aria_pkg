# Multimodal Data Streaming Component - aria_pkg

A Python library for handling real-time data streaming in a multimodal brick manipulation system. This component manages data streaming from the Project Aria glasses. Moreover, eye gaze tracking, intentian alignment and multi-view alignment are implemented.

## Overview

This library is one of three components in a complete multimodal manipulation system developed for a Bachelor's thesis. It specifically handles:

- Real time voice and RGB and ET image streaming
- Real-time gaze tracking via Project Aria eyetracking
- Voice command processing and ASR
- Intention Alignment
- Object detection and visual feature matching with the robotic arm
- Integration with LLM and manipulation components

## Installation

1. Clone the repository:
```bash
git clone https://github.com/joshopp/aria_pkg.git
cd aria_pkg
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Update SocialEye (eye tracking) as submodule:
```bash
cd src
git submodule update --init --recursive
```

4. Download required models:
   - Place YOLO mask model `best.pt` in `src/` directory
   - Configure SuperGlue path (see Configuration section)

## Requirements

- Ubuntu 22+
- Python 3.8+
- Project Aria glasses
- SuperGlue installation (https://github.com/magicleap/SuperGluePretrainedNetwork)
- YOLO model file (`best.pt`)
- Additional dependencies listed in `requirements.txt`


**Note**: AV and torchvision compatibility issues may occur - remove AV package if needed:
```bash
pip uninstall av
```

## Quick Start

### 1. Start the Interaction System
```bash
cd src
python3 start_interaction.py
```

### 2. Wait for Initialization
- Wait for eyetracking glasses to start streaming
- Look for terminal confirmation message
- White CV2 window should appear

### 3. Voice Interaction
Use the voice command pattern:
```
"START" + [your command] + "FINISH"
```

**Example**: "START grab this yellow brick FINISH"

### 5. Exit
Press `q` or `ESC` to quit the interaction loop


## Configuration

### Network Setup
For WiFi streaming, update firewall settings when running the interaction:
```python
# Run the pipeline like this sllow incoming UDP connections
python3 start_interaction --update_iptables
```

### Feature Matching (`feature_matching.py`)
```python
# Update SuperGlue path for local import
superglue_path = "/path/to/your/superglue"

# Change robot package IP in match_features()
def match_features():
    robot_pkg_ip = "192.168.1.100"  # Update this IP
```

### Voice Controller
```python
# Change speech recognition model if desired
model = "your-preferred-whisper-model"
```

## Core Components

### Main Scripts
- **`start_interaction.py`**: Main script - run this to start interaction
- **`gaze_processor.py`**: Handles image streaming
- **`voice_controller`**: Handles audios streaming, speech recognition and interactions with the LLM
- **`feature_matching.py`**: Handles intention alignment and multi-view alignment with SuperGlue
- **`real_time_inference`**: Computes and visualizes eye tracking
- **`aria_utils.py`**: Outsources Aria streaming and subscribing


## Benchmarking
Run benchmarking with the **`benchmark_asr`** script
### Audio Data Collection
```bash
# Save 10 test sentences of 6 seconds each
python benchmark_asr.py --record_audio
```

### ASR Performance Testing
```bash
# Benchmark all .wav files (currently 20 files per folder)
python benchmark_asr.py --benchmark_whisper
```

### Customization Options
**Test Sentences**: Modify at the top of `benchmark_asr.py`

**Models**: Change at the start of `benchmark_whisper()` function

**Audio Samples**: Modify folder paths at start of `benchmark_whisper()`

**Files per Folder**: Change in line 101


## Architecture
This component operates as part of a three-part system:

1. Multimodal data component (this package): Streams and processes data from the Meta Aria glasses
2. Robot Component: Handles robot control (https://github.com/joshopp/robot_pkg)
3. LLM Component: Manages language understanding and tool calling (https://github.com/joshopp/llama_pkg)
The components communicate through a distributed architecture using ZeroMQ for efficient inter-process communication.


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Academic Context

This work was developed as part of a Bachelor's thesis on multimodal manipulation systems. If you use this work in academic research, please cite appropriately.

## Contact

**Author**: joshopp  
**Project Link**: [https://github.com/joshopp/aria_pkg](https://github.com/joshopp/aria_pkg)
