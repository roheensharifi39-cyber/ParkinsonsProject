import serial
import csv
import time
import math
from collections import deque
import matplotlib.pyplot as plt

# Use the COM port from your screenshot
PORT = "COM3"
BAUD = 115200

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

filename = "stimulated_tremor.csv"

# Store last 200 points for live graph
times = deque(maxlen=200)
motion_values = deque(maxlen=200)

csv_file = open(filename, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["time_seconds", "ax", "ay", "az", "gx", "gy", "gz", "motion_magnitude"])

start_time = time.time()

plt.ion()
fig, ax_plot = plt.subplots()

print("Recording started...")
print("Move the sensor. Press CTRL + C to stop.")

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        # Skip header line from ESP32
        if "time_ms" in line:
            continue

        parts = line.split(",")

        # We expect: time_ms, ax, ay, az, gx, gy, gz
        if len(parts) != 7:
            continue

        try:
            time_ms = float(parts[0])
            ax = float(parts[1])
            ay = float(parts[2])
            az = float(parts[3])
            gx = float(parts[4])
            gy = float(parts[5])
            gz = float(parts[6])

            current_time = time.time() - start_time

            # 3D vector magnitude:
            # motion = sqrt(ax^2 + ay^2 + az^2)
            motion = math.sqrt(ax**2 + ay**2 + az**2)

            times.append(current_time)
            motion_values.append(motion)

            writer.writerow([current_time, ax, ay, az, gx, gy, gz, motion])
            csv_file.flush()

            ax_plot.clear()
            ax_plot.plot(times, motion_values)
            ax_plot.set_title("Parkinson's Tremor Motion Monitor")
            ax_plot.set_xlabel("Time (seconds)")
            ax_plot.set_ylabel("Motion Magnitude")
            ax_plot.set_ylim(min(motion_values) - 1, max(motion_values) + 1)

            plt.pause(0.01)

        except ValueError:
            continue

except KeyboardInterrupt:
    print("Recording stopped.")

finally:
    csv_file.close()
    ser.close()
    print(f"Saved data to {filename}")