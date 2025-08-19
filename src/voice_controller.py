#! /usr/bin/env python
import csv
from faster_whisper import WhisperModel
import json
import os
import re
import sys
import zmq

import aria.sdk as aria

import gpt_api
from StreamingClientObserver import AudioObserver
from aria_utils import AriaStreamer



def stream_audio():
    # 1. load whisper model
    model = WhisperModel("base", device="auto", compute_type="int8")  # "int8" ist schnell auf CPU/GPU
    # model = WhisperModel("medium.en", device="cpu", compute_type="int8") # medium.en

    # 2. bind/connect 0mq sockets
    context = zmq.Context()
    command_socket = context.socket(zmq.PUB)
    command_socket.bind("tcp://*:5556")
    print("ZMQ bound to port 5556: publishing commands")
    avalon_socket = context.socket(zmq.REQ) 
    avalon_socket.connect("tcp://localhost:8000")  # SSH -L tunnel # IP to the Avalon1 server
    print("ZMW connected to Avalon1: Requesting GPT inference")
    llm_pub_socket = context.socket(zmq.PUSH)
    llm_pub_socket.connect("tcp://localhost:5557")
    print("ZMQ connected to port 5557: pushing inference result")

     # 3. create audio streamer and subscribe to desired data channels
    audio_streamer = AriaStreamer()
    data_channels = [aria.StreamingDataType.Audio]  
    message_size = 100  # Adjust as needed, 1 is usually sufficient for real-time applications
    observer = audio_streamer.stream_subscribe(data_channels, AudioObserver(), message_size)

    # 4. start processing audio data
    quit_flag = False
    save_flag = False
    start_time = 0
    command = "command WAIT"
    # Speak "Start" to start the recording, speak "finish" to save the command and start LLM Inference
    while not quit_flag:
        data = [["startTime_ns", "endTime_ns", "written", "confidence"]] 
        if observer.received:
            # Receive audio stream, save as audios_16k
            audios_16k, starttime_ns = observer.resample_audio()
            # transcribe with Whisper model
            segments, info = model.transcribe(
                audios_16k,
                language="en",
                word_timestamps=True,  # einzelne Wörter mit Zeit
                vad_filter=True,
                beam_size=5,
                condition_on_previous_text=False)
            
            if segments is None:
                print("No segments detected, continue listening...")
                continue
            for segment in segments:
                for word in segment.words:  
                    # print (f"[{round(word.start,2)}s, -> {round(word.end,2)}s] {word.word}")
                    normalized_word = re.sub(r"[^\w]", "", word.word.lower())
                    if normalized_word == "start": # start detected, start img interaction
                        print("START DETECTED!\n")
                        start_time = word.start
                        save_flag = True
                        command = "command START"
                    elif normalized_word == "finish" and save_flag: # end detected, end img interaction
                        print("FINISH DETECTED!\n")
                        quit_flag = True
                        save_flag = False
                        command = "command END"

                    if save_flag: # save spoken words TODO maybe starttime anpassen 
                        if word.start >= start_time:
                            s_to_ns = int(1e9)
                            begin = int(word.start * s_to_ns + starttime_ns)
                            end = int(word.end * s_to_ns + starttime_ns)
                            print(f"[{begin}ns, -> {end}ns] {word.word}")
                            data.append([begin, end, word.word, word.probability])
        
        # Send command with a topic prefix "command" cia 0mq
        command_socket.send_string(command)


    # 5. Unsubscribe to clean up resources
    print("Stop listening to audio data")
    audio_streamer.streaming_client.unsubscribe()
    command_socket.close()

    # 6. save data/word list
    print("Saving word list to CSV file...")
    csv_folder = "/home/jruopp/thesis_ws/src/aria_pkg/data/audio" # Lab
    if not os.path.exists(csv_folder):
        os.makedirs(csv_folder)
    csv_filepath = os.path.join(csv_folder, 'word_list.csv')
    if os.path.exists(csv_filepath):
        os.remove(csv_filepath)
    del data[1] # delete first row "start"
    with open(csv_filepath, mode="w") as file:
        writer = csv.writer(file)
        writer.writerows(data)

    # 7. process data and initialize GPT inference (LLama)
    question = gpt_api.combine_written_to_string(csv_filepath)
    #send 0mq request to Avalon1 server
    avalon_socket.send_string(question)
    print("starting GPT inference: Question sent to Avalon1 server...")
    response = avalon_socket.recv_json()

    # process tool call
    tool_response = json.loads(response.get("tool"))
    if tool_response is not None:
        print(f"Tool response received: <{tool_response}\n")
        tool_call = tool_response["function_name"][0]
        print(f"Tool call: {tool_call}")
        if tool_call == "grab_brick": 
            pass
        else:
            # Publish to 0mq, when not grabbing a brick no intention alignment needed
            tool_string = json.dumps(tool_response, indent=2)
            llm_pub_socket.send_string(tool_string)
            print(f"tool_call {tool_string}\n published to 0mq, ending GPT inference...")
            sys.exit()
    else:
        print("No tool response received from Avalon1 server, check if it is running")

    # process intention alignment
    intent_response = response.get("intent")
    if intent_response is not None:
        json_output = json.loads(gpt_api.extract_python_code(intent_response))
        print(f"Intent response received: {json_output}\n")
        intent_json = gpt_api.process_csv_and_find_timestamps(csv_filepath, question, json_output)
        tool_response["arguments"] = [intent_json]
        tool_string = json.dumps(tool_response, indent=2)
        llm_pub_socket.send_string(tool_string)
        print(f"tool_call {tool_string} published to 0mq, ending GPT inference...")
    else:
        print("No intent response received from Avalon1 server, check if it is running")

    # close 0mq sockets
    avalon_socket.close()
    llm_pub_socket .close()