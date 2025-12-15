from scipy.fft import rfft, rfftfreq
import librosa
import numpy as np
import pandas as pd

def extract_features_v2(signal: np.ndarray) -> dict:
    fft_values = np.abs(rfft(signal))
    fft_freqs = rfftfreq(len(signal), 1 / 6000)
    mfcc = librosa.feature.mfcc(y=signal, sr=6000, n_mfcc=13)

    return fft_values.tolist() + mfcc.flatten().tolist()


def build_feature_df_v2(df_in: pd.DataFrame) -> pd.DataFrame: 
    rows = [] 
    for row in df_in.itertuples(index=False): 
        signal = np.asarray(row, dtype=float) 
        feats = extract_features_v2(signal) 
        rows.append(feats)
        
    return pd.DataFrame(rows)
