import serial
import numpy as np
import struct
import time
from serial.tools.list_ports import comports
from timeit import default_timer as timer
from datetime import timedelta


portName = 'COM6'
baudRate = 115200


try:
    ser = serial.Serial()
    ser.port = portName
    ser.baudrate = baudRate
    ser.open()
    print("Opening port " + ser.name)

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


send_values = 3.0012

start_time = timer()


# wait until we have an '!' from the Arduino
bytes = ser.readline()

# decode bytes into string
received = bytes.decode('utf-8')
received = received.replace('\r', '').replace('\n', '')


if '!' in received is False:
    print("Invalid request received. Continue to wait.")
    exit(-1)


send = np.array([1.111,2.222,3.333,4.444])

for i in range(3):
    print(i)
    send_values = send * i
    print("Expected: ", np.prod(send_values))
    for j in range(len(send_values)):
        as_short = float(send_values[j])
        float_format = ">f"

        float_bytes =  bytearray(struct.pack(float_format, as_short))
        curr_time = timer()
        print( timedelta(seconds=curr_time - start_time), ": Base 10: ", as_short, "Base 16:",  hex(float_bytes[3]), hex(float_bytes[2]), hex(float_bytes[1]), hex(float_bytes[0]), "len:", len(float_bytes))


        
        # clear the input buffer in case we've been spammed
        ser.reset_input_buffer()

        # send bytes down to Arduino
        ser.write(float_bytes)

        while True:

            bytes = ser.readline()

            # decode bytes into string
            response = bytes.decode('utf-8')
            response = response.replace('\r', '').replace('\n', '')
            curr_time = timer()
            
            print(timedelta(seconds=curr_time - start_time) , ": Target reported: " , response)

            if ("!" in response or "$" in response):
                break
            time.sleep(0.01)
    
    while True:
        bytes = ser.readline()

        # decode bytes into string
        response = bytes.decode('utf-8')
        response = response.replace('\r', '').replace('\n', '')
        curr_time = timer()
        
        print(timedelta(seconds=curr_time - start_time) , ": Target reported: " , response)

        if ("%" in response):
            break
        time.sleep(0.01)
        