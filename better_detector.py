import pandas as pd
import numpy as np

files = [
    "rest_normal.csv",
    "stimulated_tremor.csv",
    "normal_movement.csv"
]

def fft_features(signal, time):
    signal = signal - np.mean(signal)

    # Windowing reduces fake FFT spikes from sharp starts/stops
    window = np.hanning(len(signal))
    signal = signal * window

    dt = np.median(np.diff(time))
    sample_rate = 1 / dt

    freqs = np.fft.rfftfreq(len(signal), d=dt)
    fft_values = np.abs(np.fft.rfft(signal))

    # Ignore 0 Hz
    valid = freqs > 0
    freqs_valid = freqs[valid]
    fft_valid = fft_values[valid]

    peak_index = np.argmax(fft_valid)
    peak_freq = freqs_valid[peak_index]
    peak_strength = fft_valid[peak_index]

    total_power = np.sum(fft_valid)

    band_4_6 = (freqs_valid >= 4) & (freqs_valid <= 6)
    band_8_12 = (freqs_valid >= 8) & (freqs_valid <= 12)

    power_4_6 = np.sum(fft_valid[band_4_6])
    power_8_12 = np.sum(fft_valid[band_8_12])

    ratio_4_6 = power_4_6 / total_power if total_power != 0 else 0
    ratio_8_12 = power_8_12 / total_power if total_power != 0 else 0

    return peak_freq, peak_strength, ratio_4_6, ratio_8_12

def analyze_file(filename):
    df = pd.read_csv(filename)

    time = df["time_seconds"].values

    axes = {
        "ax": df["ax"].values,
        "ay": df["ay"].values,
        "az": df["az"].values,
        "gx": df["gx"].values,
        "gy": df["gy"].values,
        "gz": df["gz"].values,
    }

    # Motion magnitude formula comes from 3D Pythagorean theorem:
    # magnitude = sqrt(x^2 + y^2 + z^2)
    accel_mag = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)
    accel_signal = accel_mag - np.mean(accel_mag)
    rms_motion = np.sqrt(np.mean(accel_signal**2))

    gyro_mag = np.sqrt(df["gx"]**2 + df["gy"]**2 + df["gz"]**2)
    gyro_signal = gyro_mag - np.mean(gyro_mag)
    gyro_rms = np.sqrt(np.mean(gyro_signal**2))

    best_axis = None
    best_score = -1
    best_peak_freq = 0
    best_ratio_4_6 = 0
    best_ratio_8_12 = 0

    for axis_name, signal in axes.items():
        peak_freq, peak_strength, ratio_4_6, ratio_8_12 = fft_features(signal, time)

        # Tremor score checks both the real tremor band and possible doubled harmonic band
        tremor_score = ratio_4_6 + ratio_8_12

        if tremor_score > best_score:
            best_score = tremor_score
            best_axis = axis_name
            best_peak_freq = peak_freq
            best_ratio_4_6 = ratio_4_6
            best_ratio_8_12 = ratio_8_12

    return {
        "file": filename,
        "rms_motion": rms_motion,
        "gyro_rms": gyro_rms,
        "best_axis": best_axis,
        "best_peak_freq": best_peak_freq,
        "ratio_4_6": best_ratio_4_6,
        "ratio_8_12": best_ratio_8_12,
        "tremor_score": best_ratio_4_6 + best_ratio_8_12
    }

for filename in files:
    result = analyze_file(filename)

    rms = result["rms_motion"]
    gyro = result["gyro_rms"]
    freq = result["best_peak_freq"]
    ratio_4_6 = result["ratio_4_6"]
    ratio_8_12 = result["ratio_8_12"]
    tremor_score = result["tremor_score"]

    print("\n==============================")
    print(f"File: {result['file']}")
    print(f"RMS motion: {rms:.2f}")
    print(f"Gyro RMS: {gyro:.2f}")
    print(f"Best tremor axis: {result['best_axis']}")
    print(f"Best axis peak frequency: {freq:.2f} Hz")
    print(f"4-6 Hz ratio: {ratio_4_6:.3f}")
    print(f"8-12 Hz harmonic ratio: {ratio_8_12:.3f}")
    print(f"Tremor score: {tremor_score:.3f}")

    if rms < 300 and gyro < 300:
        classification = "Resting / no major movement"

    elif rms > 2200 or gyro > 5000:
        classification = "Normal large movement"

    elif 300 <= rms <= 2200 and gyro <= 3000 and (
        ratio_4_6 > 0.10 or ratio_8_12 > 0.12 or 8 <= freq <= 12
    ):
        classification = "Tremor-like rhythmic movement"

    else:
        classification = "Movement detected, but not clearly tremor-like"

    print(f"Classification: {classification}")