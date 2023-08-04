import serial


port = serial.Serial("COM3",baudrate=115200)
port.flush()

port.write(b'r10')

port.close()