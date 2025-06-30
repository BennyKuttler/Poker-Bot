# model_calibration.py
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

def calibrate_model(clf, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    calibrated = CalibratedClassifierCV(clf, method='isotonic', cv='prefit')
    calibrated.fit(X_val, y_val)
    return calibrated