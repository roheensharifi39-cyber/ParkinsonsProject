import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

files = [
    "rest_normal.csv",
    "stimulated_tremor.csv",
    "normal_movement.csv"
]

def analyze_file(filename):
    df = pd.read_csv(filename)

    time = df["time_seconds"].values
    ax = df["ax"].values
    ay = df["ay"].values
    az = df["az"].values
    gx = df["gx"].values
    gy = df["gy"].values
    gz = df["gz"].values

    # 3D vector magnitude from Pythagorean theorem:
    # magnitude = sqrt(x^2 + y^2 + z^2)
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Remove gravity / baseline offset
    accel_signal = accel_mag - np.mean(accel_mag)
    gyro_signal = gyro_mag - np.mean(gyro_mag)

    dt = np.median(np.diff(time))
    sample_rate = 1 / dt

    freqs = np.fft.rfftfreq(len(accel_signal), d=dt)
    fft_values = np.abs(np.fft.rfft(accel_signal))

    # Ignore 0 Hz because that is just baseline/DC offset
    peak_index = np.argmax(fft_values[1:]) + 1
    peak_freq = freqs[peak_index]
    peak_strength = fft_values[peak_index]

    tremor_band = (freqs >= 4) & (freqs <= 6)

    tremor_power = np.sum(fft_values[tremor_band])
    total_power = np.sum(fft_values[1:])

    if total_power == 0:
        tremor_ratio = 0
    else:
        tremor_ratio = tremor_power / total_power

    rms_motion = np.sqrt(np.mean(accel_signal**2))
    std_motion = np.std(accel_signal)
    gyro_rms = np.sqrt(np.mean(gyro_signal**2))

    return {
        "file": filename,
        "sample_rate": sample_rate,
        "rms_motion": rms_motion,
        "std_motion": std_motion,
        "gyro_rms": gyro_rms,
        "peak_freq": peak_freq,
        "peak_strength": peak_strength,
        "tremor_ratio_4_6Hz": tremor_ratio
    }

results = []

for file in files:
    results.append(analyze_file(file))

results_df = pd.DataFrame(results)

print("\n===== Dataset Comparison =====")
print(results_df.to_string(index=False))

print("\n===== Simple Interpretation =====")

for result in results:
    file = result["file"]
    peak_freq = result["peak_freq"]
    tremor_ratio = result["tremor_ratio_4_6Hz"]
    rms_motion = result["rms_motion"]

    print(f"\nFile: {file}")
    print(f"Peak frequency: {peak_freq:.2f} Hz")
    print(f"4-6 Hz tremor ratio: {tremor_ratio:.3f}")
    print(f"RMS motion: {rms_motion:.2f}")

    if 4 <= peak_freq <= 6 and tremor_ratio > 0.20:
        print("Result: Tremor-like pattern detected")
    else:
        print("Result: Not clearly tremor-like")

# Bar chart: tremor ratio comparison
plt.figure()
plt.bar(results_df["file"], results_df["tremor_ratio_4_6Hz"])
plt.title("Tremor Band Ratio Comparison")
plt.xlabel("Dataset")
plt.ylabel("4-6 Hz Tremor Band Ratio")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Bar chart: RMS motion comparison
plt.figure()
plt.bar(results_df["file"], results_df["rms_motion"])
plt.title("Motion Intensity Comparison")
plt.xlabel("Dataset")
plt.ylabel("RMS Motion")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()
