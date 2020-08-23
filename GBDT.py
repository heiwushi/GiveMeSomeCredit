from sklearn.ensemble import GradientBoostingClassifier


def call(train_input, train_label,validate_input, validate_label):
    gbc_clf = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=8, random_state=112)
    gbc_clf.fit(train_input, train_label)
    gbc_clf_proba = gbc_clf.predict_proba(validate_input)
    gbc_clf_scores = gbc_clf_proba[:, 1]
    return gbc_clf_scores

