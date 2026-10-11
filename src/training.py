"""Four simple classifiers; select using training-only cross-validation."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def purge_overlaps(metadata, train_index, test_index):
    """Keep training windows that share no sample interval with test windows."""
    train = metadata.loc[train_index]
    test = metadata.loc[test_index]
    overlaps = ((train.start_s.to_numpy()[:, None] < test.end_s.to_numpy()) &
                (train.end_s.to_numpy()[:, None] > test.start_s.to_numpy()))
    return np.asarray(train_index)[~overlaps.any(axis=1)]


def train_models(X_train, y_train, metadata=None):
    if y_train.value_counts().min() < 3 or y_train.nunique() != 2:
        raise ValueError('Need at least three training windows in each class.')
    models = {
        'Decision Tree': DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=8,
                                                class_weight='balanced', random_state=42, n_jobs=-1),
        'KNN': make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        'SVM': make_pipeline(StandardScaler(), SVC(class_weight='balanced', C=1)),
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    splits = list(cv.split(X_train, y_train))
    if metadata is not None:
        # sklearn CV uses positions, not the original DataFrame index.
        positional = metadata.reset_index(drop=True)
        splits = [(purge_overlaps(positional, train, valid), valid) for train, valid in splits]
    for train, valid in splits:
        if y_train.iloc[train].nunique() != 2:
            raise ValueError('A purged training fold contains only one class.')
    rows = []
    for name, model in models.items():
        scores = cross_val_score(model, X_train, y_train, cv=splits, scoring='f1', error_score='raise', n_jobs=1)
        rows.append({'Model': name, 'CV F1': scores.mean()})
        model.fit(X_train, y_train)
    scores = pd.DataFrame(rows).set_index('Model')
    best_name = scores['CV F1'].idxmax()
    return models, scores, best_name
