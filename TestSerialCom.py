# testing serial com

import serial #import library
port = serial.Serial('COM3',baudrate=115200) #assign port
port.write,str.encode('r') # write a string to port, just trying this