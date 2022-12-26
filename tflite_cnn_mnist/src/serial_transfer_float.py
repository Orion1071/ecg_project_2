    # Dr Forsyth python

import serial
import pickle
import numpy as np
import time
import matplotlib.pyplot as plt
import struct
from serial.tools.list_ports import comports
from timeit import default_timer as timer
from datetime import timedelta

X = np.load("/Users/sanda/Documents/esp_dev_files/ecg_project_2/array_passing_test/X_data.npy")

# set parameters for serial port
portName='COM5'
#portName = 'COM3' 115200
baudRate = 9600

# attempt to open port
try:
    ser = serial.Serial()
    ser.port = portName
    ser.baudrate = baudRate
    ser.open()
    print("Opening port " + ser.name)

# if fail, print a helpful message
except:
    print("Couldn't open port. Try changing portName to one of the options below:")
    ports_list = comports()
    for port_candidate in ports_list:
        print(port_candidate.device)
    exit(-1)

if ser.is_open == True:
    print("Success!")

else:
    print("Unable to open port :(")
    exit(-1)
    
#float64 double test ()
#revere the byte order in cpp
send_values = 3.0012




# wait until we have an '!' from the Arduino
bytes = ser.readline()

# decode bytes into string
received = bytes.decode('utf-8')
received = received.replace('\r', '').replace('\n', '')

if '!' in received is False:
    print("Invalid request received. Continue to wait.")
    exit(-1)

send_values = X.flatten()
start_main = timer()

for a in range(len(send_values)):

    as_short = float(send_values[a])

    # set the format to be little endian signed short (2 bytes) (int8)
    short_format = ">f"

    # convert to desired bytes size (2 byte signed int) and order
    # https://docs.python.org/3/library/struct.html
    short_bytes = bytearray(struct.pack(short_format, as_short))

    #hex(short_bytes[7]), hex(short_bytes[6]), hex(short_bytes[5]), hex(short_bytes[4]),
    print("Base 10: ", as_short, "Base 16:",  hex(short_bytes[3]), hex(short_bytes[2]), hex(short_bytes[1]), hex(short_bytes[0]), "len:", len(short_bytes))


    # clear the input buffer in case we've been spammed
    ser.reset_input_buffer()

    # send bytes down to Arduino
    ser.write(short_bytes)

    for i in range(0,7):
        # wait until a response from the board
        bytes = ser.readline()

        # decode bytes into string
        response = bytes.decode('utf-8')
        response = response.replace('\r', '').replace('\n', '')

        print("Target reported: "+response)


end_main = timer()
print(f'main done in: {timedelta(seconds=end_main - start_main)}\n')


