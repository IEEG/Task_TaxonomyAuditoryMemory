import serial

port = serial.Serial("COM3",baudrate = 115200)
port.open()
port.write(str.encode('r10'))