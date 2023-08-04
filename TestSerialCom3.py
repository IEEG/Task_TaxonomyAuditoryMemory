from serial import Serial
from time import sleep


def send(message):
    ser = Serial("COM3", baudrate=115200)
    ser.flush()
    ser.write(message.encode())
    sleep(0.1)
    if ser.inWaiting():
        print(ser.readline().decode().rstrip('\r\n').split(",")[0])