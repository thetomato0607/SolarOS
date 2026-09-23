"""Chronological train/validation/test split shared by notebooks 03-05."""

TRAIN_END = "2018-08-01"   # train: Jan-Jul
TEST_START = "2018-10-01"  # validation: Aug-Sep; test: Oct-Dec


def chronological_split(X, y):
    """Split by timestamp so no future hour leaks into training.

    Returns:
        ``(X_train, y_train, X_val, y_val, X_test, y_test)``.
    """
    train_mask = X.index < TRAIN_END
    val_mask = (X.index >= TRAIN_END) & (X.index < TEST_START)
    test_mask = X.index >= TEST_START

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    assert X_train.index.max() < X_val.index.min(), "Train bleeds into val"
    assert X_val.index.max() < X_test.index.min(), "Val bleeds into test"
    return X_train, y_train, X_val, y_val, X_test, y_test
