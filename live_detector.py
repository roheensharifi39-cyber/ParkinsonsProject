import serial
import time
import numpy as np
from collections import deque

PORT = "COM3"
BAUD = 115200

WINDOW_SECONDS = 5
PRINT_EVERY_SECONDS = 1.5

ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# Store live data
times = deque()
ax_data = deque()
ay_data = deque()
az_data = deque()
gx_data = deque()
gy_data = deque()
gz_data = deque()

def fft_features(signal, time_values):
    signal = np.array(signal)
    time_values = np.array(time_values)

    signal = signal - np.mean(signal)

    if len(signal) < 20:
        return 0, 0, 0

    # Windowing reduces fake FFT spikes
    window = np.hanning(len(signal))
    signal = signal * window

    dt = np.median(np.diff(time_values))

    if dt <= 0:
        return 0, 0, 0

    freqs = np.fft.rfftfreq(len(signal), d=dt)
    fft_values = np.abs(np.fft.rfft(signal))

    valid = freqs > 0
    freqs_valid = freqs[valid]
    fft_valid = fft_values[valid]

    if len(fft_valid) == 0:
        return 0, 0, 0

    peak_index = np.argmax(fft_valid)
    peak_freq = freqs_valid[peak_index]

    total_power = np.sum(fft_valid)

    band_4_6 = (freqs_valid >= 4) & (freqs_valid <= 6)
    band_8_12 = (freqs_valid >= 8) & (freqs_valid <= 12)

    power_4_6 = np.sum(fft_valid[band_4_6])
    power_8_12 = np.sum(fft_valid[band_8_12])

    ratio_4_6 = power_4_6 / total_power if total_power != 0 else 0
    ratio_8_12 = power_8_12 / total_power if total_power != 0 else 0

    return peak_freq, ratio_4_6, ratio_8_12

def classify_motion():
    time_values = np.array(times)

    ax = np.array(ax_data)
    ay = np.array(ay_data)
    az = np.array(az_data)
    gx = np.array(gx_data)
    gy = np.array(gy_data)
    gz = np.array(gz_data)

    # Acceleration magnitude comes from 3D Pythagorean theorem:
    # magnitude = sqrt(x^2 + y^2 + z^2)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    accel_signal = accel_mag - np.mean(accel_mag)
    rms_motion = np.sqrt(np.mean(accel_signal**2))

    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_signal = gyro_mag - np.mean(gyro_mag)
    gyro_rms = np.sqrt(np.mean(gyro_signal**2))

    axes = {
        "ax": ax,
        "ay": ay,
        "az": az,
        "gx": gx,
        "gy": gy,
        "gz": gz
    }

    best_axis = None
    best_score = -1
    best_peak_freq = 0
    best_ratio_4_6 = 0
    best_ratio_8_12 = 0

    for axis_name, signal in axes.items():
        peak_freq, ratio_4_6, ratio_8_12 = fft_features(signal, time_values)

        tremor_score = ratio_4_6 + ratio_8_12

        if tremor_score > best_score:
            best_score = tremor_score
            best_axis = axis_name
            best_peak_freq = peak_freq
            best_ratio_4_6 = ratio_4_6
            best_ratio_8_12 = ratio_8_12

    if rms_motion < 300 and gyro_rms < 300:
        classification = "RESTING / NO MAJOR MOVEMENT"

    elif rms_motion > 2200 or gyro_rms > 5000:
        classification = "NORMAL LARGE MOVEMENT"

    elif 300 <= rms_motion <= 2200 and gyro_rms <= 3000 and (
        best_ratio_4_6 > 0.10 or best_ratio_8_12 > 0.12 or 8 <= best_peak_freq <= 12
    ):
        classification = "TREMOR-LIKE RHYTHMIC MOVEMENT"

    else:
        classification = "MOVEMENT DETECTED, NOT CLEARLY TREMOR-LIKE"

    return classification, rms_motion, gyro_rms, best_axis, best_peak_freq, best_ratio_4_6, best_ratio_8_12, best_score

print("Live Parkinson's Tremor Detector started.")
print("Keep hand still, simulate tremor, or move normally.")
print("Press CTRL + C to stop.\n")

last_print_time = time.time()

try:
    while True:
        line = ser.readline().decode(errors="ignore").strip()

        if not line:
            continue

        if "time_ms" in line:
            continue

        parts = line.split(",")

        if len(parts) != 7:
            continue

        try:
            t = float(parts[0]) / 1000.0
            ax = float(parts[1])
            ay = float(parts[2])
            az = float(parts[3])
            gx = float(parts[4])
            gy = float(parts[5])
            gz = float(parts[6])
        except ValueError:
            continue

        times.append(t)
        ax_data.append(ax)
        ay_data.append(ay)
        az_data.append(az)
        gx_data.append(gx)
        gy_data.append(gy)
        gz_data.append(gz)

        # Keep only last WINDOW_SECONDS of data
        while len(times) > 0 and times[-1] - times[0] > WINDOW_SECONDS:
            times.popleft()
            ax_data.popleft()
            ay_data.popleft()
            az_data.popleft()
            gx_data.popleft()
            gy_data.popleft()
            gz_data.popleft()

        if time.time() - last_print_time >= PRINT_EVERY_SECONDS and len(times) > 80:
            result = classify_motion()

            classification = result[0]
            rms_motion = result[1]
            gyro_rms = result[2]
            best_axis = result[3]
            best_peak_freq = result[4]
            ratio_4_6 = result[5]
            ratio_8_12 = result[6]
            tremor_score = result[7]

            print("\n==============================")
            print(f"Classification: {classification}")
            print(f"RMS motion: {rms_motion:.2f}")
            print(f"Gyro RMS: {gyro_rms:.2f}")
            print(f"Best axis: {best_axis}")
            print(f"Peak frequency: {best_peak_freq:.2f} Hz")
            print(f"4-6 Hz ratio: {ratio_4_6:.3f}")
            print(f"8-12 Hz harmonic ratio: {ratio_8_12:.3f}")
            print(f"Tremor score: {tremor_score:.3f}")

            last_print_time = time.time()

except KeyboardInterrupt:
    print("\nStopped live detector.")

finally:
    ser.close()