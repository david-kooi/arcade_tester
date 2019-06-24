import zmq
import time


context = zmq.Context()
socket = context.socket(zmq.PUB)
socket.bind("tcp://*:1111")

while True:
    socket.send_string("Hello")
    time.sleep(1)
