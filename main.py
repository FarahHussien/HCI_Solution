import imports as _import_

ali_signal = 'D:\Materials FCIS\Year 4\Semester 8\HCI\HCI_Solution\Data\ECG_Ali.txt'
mohamed_signal = 'D:\Materials FCIS\Year 4\Semester 8\HCI\HCI_Solution\Data\ECG_Mohamed.txt'
test_signal = 'D:\Materials FCIS\Year 4\Semester 8\HCI\HCI_Solution\Data\Test signal.txt'

x_01, y_01= _import_.reader.read_ecg_data(ali_signal)
x_02, y_02= _import_.reader.read_ecg_data(mohamed_signal)

filtered_signal_01 = _import_.reader.butter_bandpass_filter(y_01,low_cutoff=0.5,high_cutoff=40,fs=250,order=2)
filtered_signal_02 = _import_.reader.butter_bandpass_filter(y_02,low_cutoff=0.5,high_cutoff=40,fs=250,order=2)

# derivative_02, dy_02 = _import_.reader.apply_derivative(x_02, filtered_signal_02)
# result = [dy_02[i]**2 for i in range(len(dy_02))] # square el signal
# win_size = round(0.03 * 250)
# sum = 0
# # apply moving avg filter to smooth squared signal
# for j in range(win_size):
#     sum += result[j] / win_size
#     result[j] = sum
# for index in range(win_size, len(result)):
#     sum += result[index] / win_size
#     sum -= result[index - win_size] / win_size
#     result[index] = sum

# ali sig - part01
peaks_01, _ = _import_.find_peaks(filtered_signal_01) # find all peaks in signal
X, Y = [] , []
for i in range(len(peaks_01)-1):
    L = peaks_01[i]
    Y.append(filtered_signal_01[L])
    X.append(filtered_signal_01[i])

peak_value_01 = max(Y)  # Highest amplitude peak

threshold = 0.6 * peak_value_01
R_peaks_01, X_Rpos_01 = [] , []

# identify R-peaks
for i in range(len(peaks_01)-1):
    L2 = peaks_01[i]
    if filtered_signal_01[L2] > threshold:
        R_peaks_01.append(filtered_signal_01[L2])
        X_Rpos_01.append(peaks_01[i])
        
N = 1000
samples = range(N)

_import_.plt.figure(figsize=(10,4))
_import_.plt.plot(samples, filtered_signal_01[:N], 'b-', label='Filtered ECG')

# identify P-peaks
P_peaks_01, X_Ppos_01 = [] , []

lookback_samples = int(0.2 * 250)  # 200ms at 250Hz

for r_pos in X_Rpos_01:
    if r_pos - lookback_samples >= 0:
        search_segment = filtered_signal_01[r_pos - lookback_samples:r_pos - 20]  # avoid overlap with QRS
        if len(search_segment) > 0:
            local_max_index = _import_.np.argmax(search_segment)
            p_pos = r_pos - lookback_samples + local_max_index
            P_peaks_01.append(filtered_signal_01[p_pos])
            X_Ppos_01.append(p_pos)

# identify T-peaks
T_peaks_01, X_Tpos_01 = [] , []

# Look forward around 300ms (75 samples) after each R peak
lookahead_start = int(0.15 * 250)  # 150ms
lookahead_end = int(0.3 * 250)     # 300ms

for r_pos in X_Rpos_01:
    start = r_pos + lookahead_start
    end = r_pos + lookahead_end

    if end < len(filtered_signal_01):
        search_segment = filtered_signal_01[start:end]
        if len(search_segment) > 0:
            local_max_index = _import_.np.argmax(search_segment)
            t_pos = start + local_max_index
            T_peaks_01.append(filtered_signal_01[t_pos])
            X_Tpos_01.append(t_pos)

# plotting signal peaks
r_in_window = [x for x in X_Rpos_01 if x < N]
r_amps = [R_peaks_01[i] for i, x in enumerate(X_Rpos_01) if x < N]
_import_.plt.plot(r_in_window, r_amps, 'ro', markersize=8, label='R-peaks')

p_in_window = [x for x in X_Ppos_01 if x < N]
p_amps = [P_peaks_01[i] for i, x in enumerate(X_Ppos_01) if x < N]
_import_.plt.plot(p_in_window, p_amps, 'go', markersize=8, label='P-peaks')

t_in_window = [x for x in X_Tpos_01 if x < N]
t_amps = [T_peaks_01[i] for i, x in enumerate(X_Tpos_01) if x < N]
_import_.plt.plot(t_in_window, t_amps, 'yo', markersize=8, label='T-peaks')

_import_.plt.title(f"First {N} Samples with R-peaks & P-peaks & T-peaks Detection in Ali's Signal")
_import_.plt.xlabel("Samples")
_import_.plt.ylabel("Amplitude")
_import_.plt.legend()
_import_.plt.grid(True)

_import_.plt.show()

# mohamed sig - part02
peaks_02, _ = _import_.find_peaks(filtered_signal_02) # find all peaks in signal
X, Y = [] , []
for i in range(len(peaks_02)-1):
    L = peaks_02[i]
    Y.append(filtered_signal_02[L])
    X.append(filtered_signal_02[i])

peak_value_02 = max(Y)  # Highest amplitude peak

threshold = 0.6 * peak_value_02
R_peaks_02, X_Rpos_02 = [] , []

# identify R-peaks
for i in range(len(peaks_02)-1):
    L2 = peaks_02[i]
    if filtered_signal_02[L2] > threshold:
        R_peaks_02.append(filtered_signal_02[L2])
        X_Rpos_02.append(peaks_02[i])
        
N = 1000
samples = range(N)

_import_.plt.figure(figsize=(10,4))
_import_.plt.plot(samples, filtered_signal_02[:N], 'b-', label='Filtered ECG')

# identify P-peaks
P_peaks_02, X_Ppos_02 = [] , []

lookback_samples = int(0.2 * 250)  # 200ms at 250Hz

for r_pos in X_Rpos_02:
    if r_pos - lookback_samples >= 0:
        search_segment = filtered_signal_02[r_pos - lookback_samples:r_pos - 20]  # avoid overlap with QRS
        if len(search_segment) > 0:
            local_max_index = _import_.np.argmax(search_segment)
            p_pos = r_pos - lookback_samples + local_max_index
            P_peaks_02.append(filtered_signal_02[p_pos])
            X_Ppos_02.append(p_pos)

# identify T-peaks
T_peaks_02, X_Tpos_02 = [] , []

# Look forward around 300ms (75 samples) after each R peak
lookahead_start = int(0.15 * 250)  # 150ms
lookahead_end = int(0.3 * 250)     # 300ms

for r_pos in X_Rpos_02:
    start = r_pos + lookahead_start
    end = r_pos + lookahead_end

    if end < len(filtered_signal_02):
        search_segment = filtered_signal_02[start:end]
        if len(search_segment) > 0:
            local_max_index = _import_.np.argmax(search_segment)
            t_pos = start + local_max_index
            T_peaks_02.append(filtered_signal_02[t_pos])
            X_Tpos_02.append(t_pos)

# plotting signal peaks
r_in_window = [x for x in X_Rpos_02 if x < N]
r_amps = [R_peaks_02[i] for i, x in enumerate(X_Rpos_02) if x < N]
_import_.plt.plot(r_in_window, r_amps, 'ro', markersize=8, label='R-peaks')

p_in_window = [x for x in X_Ppos_02 if x < N]
p_amps = [P_peaks_02[i] for i, x in enumerate(X_Ppos_02) if x < N]
_import_.plt.plot(p_in_window, p_amps, 'go', markersize=8, label='P-peaks')

t_in_window = [x for x in X_Tpos_02 if x < N]
t_amps = [T_peaks_02[i] for i, x in enumerate(X_Tpos_02) if x < N]
_import_.plt.plot(t_in_window, t_amps, 'yo', markersize=8, label='T-peaks')

_import_.plt.title(f"First {N} Samples with R-peaks & P-peaks & T-peaks Detection in Mohammed's Signal")
_import_.plt.xlabel("Samples")
_import_.plt.ylabel("Amplitude")
_import_.plt.legend()
_import_.plt.grid(True)
_import_.plt.show()

# apply dwt on ali's sig
dwt_coeffs_01, filtered_01 = _import_.reader.extract_dwt_features(y_01)
dwt_coeffs_02, filtered_02 = _import_.reader.extract_dwt_features(y_02)

# print("dwt_coeffs_01: ", dwt_coeffs_01[0])
# print(filtered_01[:5])
# print("dwt_coeffs_02: ", dwt_coeffs_02[0])

# apply ac/dct on signals 
dct_coeffs_01 = _import_.reader.extract_ac_dct_features(x_01)
dct_coeffs_02 = _import_.reader.extract_ac_dct_features(x_02)

# print(dct_coeffs_01[:])
# print(dct_coeffs_02[:5])

# extract ECG features - ali's sig - mohammed's sig
ali_features = _import_.reader.extract_ecg_features(P_peaks_01, X_Ppos_01, R_peaks_01, X_Rpos_01, T_peaks_01, X_Tpos_01, dct_coeffs_01[:], filtered_01[:])
mohamed_features = _import_.reader.extract_ecg_features(P_peaks_02, X_Ppos_02, R_peaks_02, X_Rpos_02, T_peaks_02, X_Tpos_02, dct_coeffs_02[:], filtered_02[:])

ali_features.to_csv("D:\Materials FCIS\Year 4\Semester 8\HCI\HCI_Solution\Features Map\Ali_feature_map.csv", index=False)
mohamed_features.to_csv("D:\Materials FCIS\Year 4\Semester 8\HCI\HCI_Solution\Features Map\Mohamed_feature_map.csv", index=False)

# print(ali_features.head())
# print(mohamed_features.head())


# test signal
x_test, y_test= _import_.reader.read_ecg_data(test_signal)
filtered_signal_test = _import_.reader.butter_bandpass_filter(y_test,low_cutoff=0.5,high_cutoff=40,fs=250,order=2)
dwt_coeffs_test, filtered_test = _import_.reader.extract_dwt_features(y_test)
dct_coeffs_test = _import_.reader.extract_ac_dct_features(x_test)

#print(dct_coeffs_test[:5])
test_features = {
    "DWT": dwt_coeffs_test,
    "DCT": dct_coeffs_test
}

result = _import_.reader.classify_signal(test_features, ali_features, mohamed_features)
print("\n\n\nThe test signal belongs to: [",result,"]\n\n\n")



# Plotting
#fig, axs = _import_.plt.subplots(1, 3, figsize=(15, 6), sharex=True, sharey=True)
# _import_.reader.plot_signal(axs, 0, X[:200], Y[:200], R_peaks[:200], 'ro', "Ali's ECG")
# _import_.plt.show()

# _import_.reader.plot_signal(axs, 0, time_part, ali_ecg_part, ali_peaks, 'ro', "Ali's ECG")
# axs[0].set_ylabel("Amplitude (v)")
# _import_.reader.plot_signal(axs, 1, time_part, mohamed_ecg_part, mohamed_peaks, 'go', "Mohamed's ECG")
# _import_.reader.plot_signal(axs, 2, time_part, unknown_ecg_part, unknown_peaks, 'bo', "Unknown ECG")

# _import_.plt.tight_layout()
# _import_.plt.show()