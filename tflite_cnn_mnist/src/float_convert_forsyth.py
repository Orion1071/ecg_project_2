import struct

import numpy as np

"""
Take an array and convert into int16
"""

print('Begin int16 test')

# create an NP array of ints
sixteen_bit_list = np.array([13, 0, -13])

# iterate through array and convert each to bytes
# 13 (base 10) = 0x000D (2's complement)
# 0 (base 10) = 0x0000 (2's complement)
# -13 (base 10) = 0xFFF3 (2's complement)
for i in sixteen_bit_list:
    # convert back to Python base types
    as_short = int(i)

    # set the format to be little endian signed short (2 bytes) (int8)
    short_format = "<h"

    # convert to desired bytes size (2 byte signed int) and order
    # https://docs.python.org/3/library/struct.html
    short_bytes = struct.pack(short_format, as_short)

    print("Base 10: ", as_short, "Base 16:", hex(short_bytes[1]), hex(short_bytes[0]))

    reconstructed_short = struct.unpack(short_format, short_bytes)[0]

    if as_short != reconstructed_short:
        print('Error re-constructing value!')


print("----------------------------")
"""
Take an array and convert into int32
"""

print('Begin int32 test')

# create a list of known values
thirtytwo_bit_list = [13, 0, -13]

# iterate through array and convert each to bytes
# 13 (base 10) = 0x0000000D (2's complement)
# 0 (base 10) = 0x00000000 (2's complement)
# -13 (base 10) = 0xFFFFFFF3 (2's complement)
for i in thirtytwo_bit_list:
    # convert back to Python base types
    as_int = int(i)

    # set the format to be little endian signed integer (4 bytes signed int) (int16)
    int_format = "<i"

    # convert to desired bytes size (4 byte signed int) and order
    # https://docs.python.org/3/library/struct.html
    int_bytes = struct.pack(int_format, as_int)

    print("Base 10: ", as_int, "Base 16:", hex(int_bytes[3]), hex(int_bytes[2]), hex(int_bytes[1]), hex(int_bytes[0]))

    reconstructed_int = struct.unpack(int_format, int_bytes)[0]

    if as_int != reconstructed_int:
        print('Error re-constructing value!')

print("----------------------------")

"""
Take an array and convert into float32
"""
print('Begin float32 test')

# pulling from Hennessy and Patterson the value of -0.75 should be:
# 0xBF40 0000 (single precision)
# 0xBFE8 0000 0000 0000 (double precision)
floats = np.array([-0.75, 0.01, 0.3])

for f in floats:
    # convert to base python type; default is double float
    # do not believe can support single precision
    as_float = float(f)

    # set the format to big endian ('>') double float ('d') (float64)
    # an alternate would be single float ('f') (float32)
    single_float_format = ">f"

    # convert to desired bytes size (8 byte single float) and order
    # https://docs.python.org/3/library/struct.html
    single_float_bytes = bytearray(struct.pack(single_float_format, as_float))

    hex_string = ""
    for b in single_float_bytes:
        hex_string += hex(b) + " "

    print("Base 10: ", as_float, "Base 16: ", hex_string)

    reconstructed_single_float = struct.unpack(single_float_format, single_float_bytes)[0]

    if abs(reconstructed_single_float - as_float) > 0.000001:
        print("Error in reconstructing values!")

print("----------------------------")


"""
Take an array and convert into float64
"""
print('Begin float64 test')

# pulling from Hennessy and Patterson (p.201) the value of -0.75 should be:
# 0xBF40 0000 (single precision)
# 0xBFE8 0000 0000 0000 (double precision)
floats = np.array([-0.75, 0.01, 0.3])

for f in floats:
    # convert to base python type; default is double float
    # do not believe can support single precision
    as_float = float(f)

    # set the format to big endian ('>') double float ('d') (float64)
    # an alternate would be single float ('f') (float32)
    double_float_format = ">d"

    # convert to desired bytes size (16 byte double float) and order
    # https://docs.python.org/3/library/struct.html
    double_float_bytes = bytearray(struct.pack(double_float_format, as_float))

    hex_string = ""
    for b in double_float_bytes:
        hex_string += hex(b) + " "

    print("Base 10: ", as_float, "Base 16: ", hex_string)

    reconstructed_single_float = struct.unpack(double_float_format, double_float_bytes)[0]

    if abs(reconstructed_single_float - as_float) > 0.000001:
        print("Error in reconstructing values!")
