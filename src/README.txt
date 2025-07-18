+Used ports with 0mq:

5555:   Req/Rep robocam image from Panda3(intel_publisher) to Aria PC(feature_matching.py)
5556:   Pub/Sub commands from Whisperer to Imaginary2
5557:   Push/Pull GPT result from Whisperer to feature_matching
5558:   Req/Rep bbox to grip from Aria PC(feature_matching) to  Panda3 (TODO: grab brick)
8000:   Req/Rep LLama interaction from Avalon1(llm_server) to Aria PC(Whisperer)

Questions:
enable color correction?
# turn on color correction
provider.set_color_correction(True)
-> maybe better for YOLO

why transformations for 2D points? Aren't they just 2D? i dont need points in 3D, right?

