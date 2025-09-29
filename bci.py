import pandas as pd
import numpy as np
import os
from scipy.signal import butter, lfilter, iirnotch
from sklearn.linear_model import LinearRegression
from mne.decoding import CSP
from sklearn.utils import resample
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# 1️⃣ Load dataset
# ----------------------------
TRAIN = r"C:\Users\intel\Documents\BCI\TRAIN"
TEST = r"C:\Users\intel\Documents\BCI\TEST"   # <-- add your test folder path here

def load_partial_csv(folder_path, chunksize=100000):
    data_list = []
    total_rows = 0
    max_rows = None
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".csv"):
                file_path = os.path.join(root, file)
                for chunk in pd.read_csv(file_path, chunksize=chunksize):
                    if max_rows is None:
                        max_rows = int(chunk.shape[0])
                    data_list.append(chunk)
                    total_rows += len(chunk)
                    if total_rows >= max_rows:
                        break
                if total_rows >= max_rows:
                    break
        if total_rows >= max_rows:
            break
    df = pd.concat(data_list, ignore_index=True)
    print(f"✅ Loaded {len(df)} rows with {df.shape[1]} columns from {folder_path}")
    return df

df_train = load_partial_csv(TRAIN)
df_test = load_partial_csv(TEST)

# ----------------------------
# 2️⃣ Filtering functions
# ----------------------------
def apply_notch(signal, fs=200, f0=50, Q=30):
    b, a = iirnotch(f0, Q, fs)
    return lfilter(b, a, signal)

def bandpass_filter(signal, fs=200, lowcut=1, highcut=40, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, signal)

eeg_channels = [col for col in df_train.columns if col not in ['Time','FeedBackEvent','EOG']]
fs = 200

# Apply filters to TRAIN
filtered_dict = {}
for ch in eeg_channels:
    filtered_dict[f"{ch}_filtered"] = bandpass_filter(apply_notch(df_train[ch], fs=fs), fs=fs)
df_train = pd.concat([df_train, pd.DataFrame(filtered_dict)], axis=1)

# Apply filters to TEST
filtered_dict_test = {}
for ch in eeg_channels:
    filtered_dict_test[f"{ch}_filtered"] = bandpass_filter(apply_notch(df_test[ch], fs=fs), fs=fs)
df_test = pd.concat([df_test, pd.DataFrame(filtered_dict_test)], axis=1)

# ----------------------------
# 3️⃣ Remove EOG artifacts
# ----------------------------
eog_cols = ['EOG']
clean_dict = {}
for ch in eeg_channels:
    model = LinearRegression().fit(df_train[eog_cols], df_train[f"{ch}_filtered"])
    cleaned = df_train[f"{ch}_filtered"] - model.predict(df_train[eog_cols])
    clean_dict[f"{ch}_clean"] = cleaned
df_train = pd.concat([df_train, pd.DataFrame(clean_dict)], axis=1)

# Apply same cleaning to TEST
clean_dict_test = {}
for ch in eeg_channels:
    model = LinearRegression().fit(df_train[eog_cols], df_train[f"{ch}_filtered"])  # use train model
    cleaned = df_test[f"{ch}_filtered"] - model.predict(df_test[eog_cols])
    clean_dict_test[f"{ch}_clean"] = cleaned
df_test = pd.concat([df_test, pd.DataFrame(clean_dict_test)], axis=1)

print("✅ Filtering + EOG artifact removal done.")

# ----------------------------
# 4️⃣ Balance dataset (train only)
# ----------------------------
class0 = df_train[df_train['FeedBackEvent'] == 0]
class1 = df_train[df_train['FeedBackEvent'] == 1]

if len(class0) > len(class1):
    class0 = resample(class0, n_samples=len(class1), random_state=42)
elif len(class1) > len(class0):
    class1 = resample(class1, n_samples=len(class0), random_state=42)

df_balanced = pd.concat([class0, class1])
print("Balanced dataset class counts:", df_balanced['FeedBackEvent'].value_counts())

# ----------------------------
# 5️⃣ CSP features
# ----------------------------
csp_channels = eeg_channels[:8]
X = df_balanced[[f"{ch}_clean" for ch in csp_channels]].values
y = df_balanced['FeedBackEvent'].values

# Reshape for CSP (n_epochs, n_channels, n_times)
X_epochs = X[:, np.newaxis, :]
X_epochs_csp = np.transpose(X_epochs, (0,2,1))

csp = CSP(n_components=4, reg=None, log=True)
X_csp_features = csp.fit_transform(X_epochs_csp, y)

df_features = pd.DataFrame(X_csp_features, columns=[f"CSP_{i+1}" for i in range(X_csp_features.shape[1])])
df_features['label'] = y
print("✅ CSP features ready for ML")

# ----------------------------
# 6️⃣ Train/test split
# ----------------------------
X = df_features.drop(columns=['label']).values
y = df_features['label'].values
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# ----------------------------
# 7️⃣ Train classifiers
# ----------------------------
svm = SVC(kernel='linear', class_weight='balanced', probability=True, random_state=42)
svm.fit(X_train, y_train)

y_pred_val = svm.predict(X_val)
print("\n🔹 Validation Results (SVM):")
print(classification_report(y_val, y_pred_val))

# ----------------------------
# 8️⃣ Create submission on TEST set
# ----------------------------
# ----------------------------
# 8️⃣ Create submission on TEST set
# ----------------------------
X_test = df_test[[f"{ch}_clean" for ch in csp_channels]].values
X_test_epochs = X_test[:, np.newaxis, :]
X_test_epochs_csp = np.transpose(X_test_epochs, (0,2,1))
X_test_csp = csp.transform(X_test_epochs_csp)

# Prediction probabilities
y_test_proba = svm.predict_proba(X_test_csp)[:, 1]

# Build submission DataFrame with synthetic IDs
submission = pd.DataFrame({
    "IdFeedBack": [f"FB{i+1:05d}" for i in range(len(y_test_proba))],
    "Prediction": y_test_proba
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv created successfully!")
