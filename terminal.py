
import serial_rx_tx
import _thread
import os
import time
import urllib.request

str_message=''
serialPort = serial_rx_tx.SerialPort()


def OpenCommand():
    
        comport = 'COM8'
        baudrate = '9600'
        serialPort.Open(comport,baudrate)
        #serialPort.Close()
            

def SendDataCommand(cmd):
    message = str(cmd)
    if serialPort.IsOpen():
        #message += '\r\n'
        serialPort.Send(message)

OpenCommand()
time.sleep(2)
while(True):
    time.sleep(0.01)
##    r_link='https://api.thingspeak.com/channels/230674/fields/1/last?api_key=36KRKRUPN6XVY02F'
##    f=urllib.request.urlopen(r_link)
##    cmd = (f.readline()).decode()
##    print("CMD:" + str(cmd))
    SendDataCommand("1")
        
        
    





