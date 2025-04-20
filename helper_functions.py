import imports as _import_

# function to read ecg signal data
def read_ecg_data(file_path):
    ecgX = []
    ecgY = []
    
    with open(file_path) as file:
        lines = file.readlines()
    
    for line in lines:
        if not line.strip():
            continue
            
        parts = line.split()
        
        if len(parts) >= 2:
            ecgX.append(int(parts[0]))
            ecgY.append(int(parts[1]))
        else:
            print(f"Warning: Line '{line.strip()}' doesn't contain enough data")
    
    return ecgX, ecgY

# function to apply band-pass filter
def butter_bandpass_filter(data, low_cutoff, high_cutoff, fs, order=4):
    nyq = 0.5 * fs
    low = low_cutoff / nyq
    high = high_cutoff / nyq
    b, a = _import_.butter(order, [low, high], btype='band')
    return _import_.filtfilt(b, a, data)

def apply_derivative(ecgX, filtered_signal):
    arr_x = _import_.np.array(ecgX)
    arr_y = _import_.np.array(filtered_signal)

    dx = _import_.np.diff(arr_x)  # Δx
    dy = _import_.np.diff(arr_y)  # Δy

    dx = _import_.np.where(dx == 0, 1e-10, dx) # avoid div by 0

    #print("dy=",dy," ","dx=",dx)
    derivative = dy / dx  # dy/dx (slope)
    #print(derivative)
    derivative_padded = _import_.np.zeros_like(arr_y)
    derivative_padded[:-1] = derivative

    #print("Derivative:", derivative_padded)
    return derivative_padded, dy

def plot_signal(axs, num, time, sig, peaks, way, title):
    axs[num].plot(time, sig)
    axs[num].plot(time[peaks], sig[peaks], way)
    axs[num].set_title(title)
    axs[num].set_xlabel("Time(s)")

def extract_ecg_features(P_peaks, X_Ppos, R_peaks, X_Rpos, T_peaks, X_Tpos, dct, dwt, fs=250):
    num_features = min(len(X_Ppos), len(X_Rpos), len(X_Tpos))

    feature_map = {
        "PR-amplitude": [],
        "RT-interval": [],
        "PT-slope": [],
        "DCT": [],
        "DWT": []
    }

    for i in range(num_features):
        P_amp = P_peaks[i]
        R_amp = R_peaks[i]
        T_amp = T_peaks[i]

        P_pos = X_Ppos[i]
        R_pos = X_Rpos[i]
        T_pos = X_Tpos[i]

        PR_amplitude = R_amp - P_amp
        RT_interval = T_pos - R_pos
        PT_slope = _import_.np.where(T_pos != P_pos, (T_amp - P_amp) / (T_pos - P_pos), 0)

        feature_map["PR-amplitude"].append(PR_amplitude)
        feature_map["RT-interval"].append(RT_interval)
        feature_map["PT-slope"].append(PT_slope)
        feature_map["DWT"].append(dwt[i])
        feature_map["DCT"].append(dct[i])

    return _import_.pd.DataFrame(feature_map)

def extract_dwt_features(signal, wavelet='db1', level=4):
    coeffs = _import_.wt.wavedec(signal, wavelet, level=level)
    filtered = _import_.wt.waverec([coeffs[0],coeffs[1],coeffs[2]], wavelet)
    
    return coeffs, filtered

def extract_ac_dct_features(signal_data, num_lags=1000):
    signal_array = _import_.np.array(signal_data)
    acf = _import_.sm.tsa.acf(signal_array, nlags=num_lags)
    dct_coeffs = _import_.dct(acf, type=2)
    
    return dct_coeffs.tolist()

def classify_signal(test_features, ali_features, mohamed_features, threshold=1.5):
    dct_test = _import_.np.array(test_features['DCT'][0]).flatten()
    dct_ali = _import_.np.array(ali_features['DCT'][0]).flatten()
    dct_mohamed = _import_.np.array(mohamed_features['DCT'][0]).flatten()
    
    max_dct_len = max(len(dct_test), len(dct_ali), len(dct_mohamed))
    dct_test = _import_.np.pad(dct_test, (0, max_dct_len - len(dct_test)), 'constant')
    dct_ali = _import_.np.pad(dct_ali, (0, max_dct_len - len(dct_ali)), 'constant')
    dct_mohamed = _import_.np.pad(dct_mohamed, (0, max_dct_len - len(dct_mohamed)), 'constant')
    
    dct_dist_ali = _import_.euc(dct_test, dct_ali)
    dct_dist_mohamed = _import_.euc(dct_test, dct_mohamed)
    
    dwt_test = _import_.np.array(test_features['DWT'][0]).flatten()
    dwt_ali = _import_.np.array(ali_features['DWT'][0]).flatten()
    dwt_mohamed = _import_.np.array(mohamed_features['DWT'][0]).flatten()
    
    max_dwt_len = max(len(dwt_test), len(dwt_ali), len(dwt_mohamed))
    dwt_test = _import_.np.pad(dwt_test, (0, max_dwt_len - len(dwt_test)), 'constant')
    dwt_ali = _import_.np.pad(dwt_ali, (0, max_dwt_len - len(dwt_ali)), 'constant')
    dwt_mohamed = _import_.np.pad(dwt_mohamed, (0, max_dwt_len - len(dwt_mohamed)), 'constant')
    
    dwt_dist_ali = _import_.euc(dwt_test, dwt_ali)
    dwt_dist_mohamed = _import_.euc(dwt_test, dwt_mohamed)
    
    total_dist_ali = dct_dist_ali + dwt_dist_ali
    total_dist_mohamed = dct_dist_mohamed + dwt_dist_mohamed
    
    min_dist = min(total_dist_ali, total_dist_mohamed)
    if min_dist > threshold * _import_.np.mean([total_dist_ali, total_dist_mohamed]):
        return "Unknown"
    
    return "Ali" if total_dist_ali < total_dist_mohamed else "Mohamed"
