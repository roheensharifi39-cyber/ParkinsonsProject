import serial
import time

PORT = "COM3"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

print("Reading raw ESP32 data. Press CTRL + C to stop.\n")

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if line:
            print(repr(line))
except KeyboardInterrupt:
    ser.close()
    print("Stopped.")