This program detects phishing websites using a Random Forest classifier, an ensemble learning method that combines multiple decision trees for better accuracy.
It begins by loading and exploring the dataset, checking for missing values, and analyzing the target class distribution.
Features are prepared by separating them from the target and encoding categorical variables if necessary.
The dataset is split into training and testing sets, and features are scaled to improve model performance.
Hyperparameter tuning with GridSearchCV optimizes the Random Forest, and evaluation is done using accuracy, classification report, and confusion matrix.
Finally, feature importance is displayed, and the trained model along with the scaler is saved for future predictions.
