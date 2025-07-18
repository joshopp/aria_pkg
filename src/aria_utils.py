import operator
from functools import reduce

import aria.sdk as aria




class AriaStreamer:
    def __init__(self):
        # Optional: Set SDK's log level to Trace or Debug for more verbose logs
        aria.set_log_level(aria.Level.Info)

        # Create DeviceClient instance
        self.device_client = aria.DeviceClient()
        self.streaming_client = aria.StreamingClient()


    def stream_start(self, device_ip, interface, profile):
        #1. set IP address if specified
        client_config = aria.DeviceClientConfig()
        if device_ip:
            client_config.ip_v4_address = device_ip
        self.device_client.set_client_config(client_config)


        # 2. Connect to the device and print status info
        device = self.device_client.connect()
        print("Device connected")
        status = device.status
        print(
            "Aria Device Status: battery level {0}, wifi ssid {1}, wifi ip {2}, mode {3}".format(
                status.battery_level, status.wifi_ssid, status.wifi_ip_address, status.device_mode
            )
        )


        # 3. Retrieve the streaming_manager
        streaming_manager = device.streaming_manager


        # 4. Set custom config for streaming
        streaming_config = aria.StreamingConfig()
        streaming_config.profile_name = profile
        if interface == "usb":
            streaming_config.streaming_interface = aria.StreamingInterface.Usb
        # Use ephemeral streaming certificates
        streaming_config.security_options.use_ephemeral_certs = True
        streaming_manager.streaming_config = streaming_config


        # 5. Start streaming and print state
        streaming_manager.start_streaming()
        streaming_state = streaming_manager.streaming_state
        print(f"Streaming state: {streaming_state}")
        print(f"Streaming Profile: {streaming_manager.streaming_config.profile_name}")

        return device


    def stream_subscribe(self, streamed_data, observer, message_size):
 
        # 1. Set costum config for subscribing
        config = self.streaming_client.subscription_config
        config.subscriber_data_type = reduce(operator.or_, streamed_data)
        #only use recent frame (1)
        for data_type in streamed_data:
            config.message_queue_size[data_type] = message_size
        

        # 2. Set the security options
        options = aria.StreamingSecurityOptions()
        options.use_ephemeral_certs = True
        config.security_options = options
        self.streaming_client.subscription_config = config


        # 3. Attach observer and subscribe start listening
        self.streaming_client.set_streaming_client_observer(observer)
        self.streaming_client.subscribe()
        print("Start listening to data")

        return observer
    

    def stream_end(self, device):
        # Unsubscribe to clean up resources
        self.streaming_client.unsubscribe()
        device.streaming_manager.stop_streaming()
        self.device_client.disconnect(device)
        print("Stopped streaming and disconnected from device")