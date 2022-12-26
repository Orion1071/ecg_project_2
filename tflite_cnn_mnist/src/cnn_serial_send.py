import serial
import numpy as np
from serial.tools.list_ports import comports
import struct
from timeit import default_timer as timer
from datetime import timedelta
import time

X_test = np.load("/Users/sanda/Documents/esp_dev_files/MNIST_data/MNIST_X_test.npy")
y_test = np.load("/Users/sanda/Documents/esp_dev_files/MNIST_data/MNIST_y_test.npy")


# set parameters for serial port
portName='COM6'
#portName = 'COM3' 115200
baudRate = 115200

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

#decode byte
received = bytes.decode('utf-8')
received = received.replace('\r', '').replace('\n', '')

if '!' in received is False:
    print("Invalid request received. Continue to wait.")
    exit(-1)

send_values = X_test[0].flatten()
actual = y_test[0]

start_time = timer()


# wait until we have an '!' from the Arduino
bytes = ser.readline()

# decode bytes into string
received = bytes.decode('utf-8')
received = received.replace('\r', '').replace('\n', '')


if '!' in received is False:
    print("Invalid request received. Continue to wait.")
    exit(-1)


# send = np.array([1.111,2.222,3.333,4.444])

for i in range(5):
    print(i)
    send_values = X_test[i].flatten()
    
    for j in range(len(send_values)):
        as_short = float(send_values[j])
        float_format = ">f"

        float_bytes =  bytearray(struct.pack(float_format, as_short))
        curr_time = timer()
        
        # print( timedelta(seconds=curr_time - start_time), ": Base 10: ", as_short, "Base 16:",  hex(float_bytes[3]), hex(float_bytes[2]), hex(float_bytes[1]), hex(float_bytes[0]), "len:", len(float_bytes))


        
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
            
            # print(timedelta(seconds=curr_time - start_time) , ": Target reported: " , response)

            if ("!" in response or "$" in response):
                break
            time.sleep(0.01)
    print("Expected Total: ", np.sum(send_values))
    print("Expected Prediction: ", y_test[i])
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
        
end_time = timer()
print(f'main done in: {timedelta(seconds=end_time- start_time)}\n')

