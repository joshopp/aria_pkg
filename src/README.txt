Req/Rep LLama interaction from Avalon1(llm_server) to Aria PC(Whisperer)

Questions:
enable color correction?
# turn on color correction
provider.set_color_correction(True)
-> maybe better for YOLO


Questions:

1. Content
	Eye Gaze Tracking figure
	
	Explain setup prompt more? 	how built, input/output
	
2. Experiments
	Llama: easy, compare to OpenAI
	Whisper: 8 standard test sentences,
		 (2 narrators (m/w) )
		 3 sound levels: Clean (>15 dB), moderate (5 dB), severe (0 dB)
		 2 errors: White Noise (computer), background chatter (from mobile near robot)
		 -> 25 samples per person
		 Relevant: latency, WER

	General setup: 3 scenarios, 2-3 commands per scenario, 10 iterations per command
			-> succesrate?
	
3. General:
	Length? Yuzhi checks
	How many copies: 1x Zell, 1x PA, Zell olo wants PDF
	2nd LLM atomic actions -> only grab brick now
	

	