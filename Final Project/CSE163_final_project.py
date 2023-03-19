'''
Vincent Liu, Zhiming Huang, Junjin Wang
CSE 163
Final Project
This python will be the main file for this final project.
In this file, we will compute serveral graphs, and do
machine learning (ML) on the dataset. You can also read
the report for a brief summary on what the graphs and outputs mean.
'''
import data_processing
import testfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV


def correlation_graph(white_wine: pd.DataFrame) -> None:
    '''
    This method takes a pandas DataFrame, and generates a heat map.
    The heat map presents the correlation between various features of
    white wine in feature space.
    '''
    features_white = white_wine.drop('quality', axis='columns')
    corr = features_white.corr()
    plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        xticklabels=corr.columns,
        yticklabels=corr.columns,
        annot=True,
        cmap='coolwarm')
    plt.tight_layout()
    plt.xticks(rotation=45)
    plt.title('Correlation Between Various Features in Feature Space')
    plt.savefig('correlation_graph.png')


def features_subplot(white_wine: pd.DataFrame) -> None:
    '''
    This method takes a pandas DataFrame, and generates 11 subplots.
    These subplots represent the relationship between each feature of
    white wine and quality of whine wine.
    '''
    plt.figure(figsize=(15, 10))
    plt.subplot(3, 4, 1)
    sns.barplot(x='quality', y='fixed_acidity', data=white_wine)
    plt.subplot(3, 4, 2)
    sns.barplot(x='quality', y='volatile_acidity', data=white_wine)
    plt.subplot(3, 4, 3)
    sns.barplot(x='quality', y='citric_acid', data=white_wine)
    plt.subplot(3, 4, 4)
    sns.barplot(x='quality', y='residual_sugar', data=white_wine)
    plt.subplot(3, 4, 5)
    sns.barplot(x='quality', y='chlorides', data=white_wine)
    plt.subplot(3, 4, 6)
    sns.barplot(x='quality', y='free_sulfur_dioxide', data=white_wine)
    plt.subplot(3, 4, 7)
    sns.barplot(x='quality', y='total_sulfur_dioxide', data=white_wine)
    plt.subplot(3, 4, 8)
    sns.barplot(x='quality', y='density', data=white_wine)
    plt.subplot(3, 4, 9)
    sns.barplot(x='quality', y='pH', data=white_wine)
    plt.subplot(3, 4, 10)
    sns.barplot(x='quality', y='sulphates', data=white_wine)
    plt.subplot(3, 4, 11)
    sns.barplot(x='quality', y='alcohol', data=white_wine)
    plt.tight_layout()
    plt.savefig('features_subplot.png')


def target_count(white_wine: pd.DataFrame) -> None:
    '''
    This method takes a pandas DataFrame, and generates a count plot.
    The count plot shows the distribution of different qualities of white
    wines in all white wines, which represent the imbalance of the dataset.
    '''
    plt.figure(figsize=(15, 10))
    sns.countplot(x='quality', data=white_wine)
    plt.savefig('target_count.png')


def balanced_target_count(white_wine: pd.DataFrame) -> None:
    '''
    This method takes a pandas DataFrame, and generates a count plot.
    1 on the X-axis means good wine. -1 on the X-axis means bad wine.
    The count plot shows the distribution of good wine and bad wine.
    Although good wines are less than bad wines, the distribution is
    more balanced.
    '''
    sns.countplot(x='wine_class', data=white_wine)
    plt.savefig('balanced_target_count.png')


def plot_scores(ax, title, search, hyperparameters, score_key) -> None:
    '''
    This method takes ax, title, search, hyperparameter, score_key and
    generates a 3D wireframe plot.
    It plots the train and validation accuracy of the models for different
    settings of the hyper-parameters. The plot is in 3D since there are 2
    inputs for each model specification.
    This method use a sklearn module that does this for us using
    k-fold validation.
    This method will never be used by users, because it serves for our
    machine_learning method.
    '''
    cv_results = search.cv_results_
    scores = cv_results[score_key]
    scores = scores.reshape(
        (len(
            hyperparameters['max_depth']), len(
            hyperparameters['min_samples_leaf'])))
    max_depths = cv_results['param_max_depth'].reshape(
        scores.shape).data.astype(int)
    min_samples_leafs = cv_results['param_min_samples_leaf'].reshape(
        scores.shape).data.astype(int)

    ax.plot_wireframe(max_depths, min_samples_leafs, scores)
    ax.view_init(20, 220)
    ax.set_xlabel('Maximum Depth')
    ax.set_ylabel('Minimum Samples Leaf')
    ax.set_zlabel('Accuracy')
    ax.set_title(title)


def machine_learning(
        white_wine: pd.DataFrame,
        normal_df: pd.DataFrame) -> None:
    '''
    In this function, we did a lot of ML related work.
    Please refer to each section for detail comments and notes.
    '''

    # we splitted data to 80% train, 10% validation, and 10% test data.
    normal_df['wine_class'] = white_wine['wine_class']
    print(normal_df)
    train_data, test_data = train_test_split(normal_df, test_size=0.2)
    validation_data, test_data = train_test_split(test_data, test_size=0.5)

    features = list(normal_df.columns)
    features.remove('wine_class')

    # It trains a dummy classifier as our baseline model and predicts the
    # target and records the accuracy rate of validation, and test data for
    # evaluating the following models.
    dummy_classifier = DummyClassifier(strategy='most_frequent')
    dummy_classifier.fit(train_data[features], train_data['wine_class'])
    val_acc_baseline = dummy_classifier.score(
        validation_data[features],
        validation_data['wine_class'])
    test_acc_baseline = dummy_classifier.score(
        test_data[features], test_data['wine_class'])
    print("Baseline Validation Accuracy = ", val_acc_baseline)
    print("Baseline Test Accuracy = ", test_acc_baseline)

    # Using GridSearch cross validation to find the best hyperperemeters
    # for the following model training purposes.
    hyperparameters = {
        'min_samples_leaf': [
            1, 10, 50, 100, 200, 300], 'max_depth': [
            1, 5, 10, 15, 20, 30]}
    search = GridSearchCV(
        DecisionTreeClassifier(),
        hyperparameters,
        cv=6,
        return_train_score=True)
    search.fit(train_data[features], train_data['wine_class'])
    params = search.best_params_
    print("Learned best max_depth:", params['max_depth'])

    # It trains a decision tree classifier model using the best parameters we
    # found. It predicts the target and records the accuracy rate of
    # validation, and test data for evaluating the following models.
    decision_tree_model = DecisionTreeClassifier(max_depth=params['max_depth'])
    decision_tree_model.fit(train_data[features], train_data['wine_class'])
    y_validation_pred = decision_tree_model.predict(validation_data[features])
    decision_validation_accuracy = accuracy_score(
        validation_data['wine_class'], y_validation_pred)
    y_test_pred = decision_tree_model.predict(test_data[features])
    decision_test_accuracy = accuracy_score(
        test_data['wine_class'], y_test_pred)
    print("DecisionTree Validation Accuracy = ", decision_validation_accuracy)
    print("DecisionTree Test Accuracy = ", decision_test_accuracy)

    # It used the method plot_scores we wrote. It plots the train and
    # validation accuracy of the models for different settings of the
    # hyper-parameters. The plot is in 3D since there are 2 inputs for each
    # model specification. This method use a sklearn module that does this for
    # us using k-fold validation.
    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    plot_scores(
        ax1,
        'Train Accuracy',
        search,
        hyperparameters,
        'mean_train_score')
    plot_scores(
        ax2,
        'Validation Accuracy',
        search,
        hyperparameters,
        'mean_test_score')
    plt.savefig('Accuracy.png')

    # It plots confusion matrix on validation and test data for visualizing
    # precision and recall rate for our trained decision tree classifier model.
    metrics.plot_confusion_matrix(
        decision_tree_model,
        validation_data[features],
        validation_data['wine_class'])
    metrics.plot_confusion_matrix(
        decision_tree_model,
        test_data[features],
        test_data['wine_class'])
    plt.show()
    print(
        metrics.accuracy_score(
            validation_data['wine_class'],
            y_validation_pred))
    plt.savefig('validation_confusion_metrics.png')
    print(metrics.accuracy_score(test_data['wine_class'], y_test_pred))
    plt.savefig('test_confusion_metrics.png')

    # It conducted feature selection using the LASSO model with normalized
    # data, identified the best parameters for our lasso_decision_tree model
    # and computed and plotted the feature importance according to the
    # decision tree classifier model, and predicted the target and recorded
    # the accuracy rate of validation, and test data for evaluating the
    # following models.
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    search = GridSearchCV(pipeline,
                          {'model__alpha': np.arange(0.1, 10, 0.1)},
                          cv=6, scoring="neg_mean_squared_error", verbose=1)
    search.fit(train_data[features], train_data['wine_class'])
    search.best_params_
    coefficients = search.best_estimator_.named_steps['model'].coef_
    importance = np.abs(coefficients)

    sns.barplot(
        y=features,
        x=importance,
        color='g')

    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features for White Wine")
    plt.legend()
    plt.show()
    plt.savefig('feature_importance.png')

    lasso_selected_feature = np.array(features)[importance > 0]
    lasso_decision_tree_model = DecisionTreeClassifier(
        max_depth=params['max_depth'])
    lasso_decision_tree_model.fit(
        train_data[lasso_selected_feature],
        train_data['wine_class'])
    y_validation_pred = lasso_decision_tree_model.predict(
        validation_data[lasso_selected_feature])
    decision_validation_accuracy = accuracy_score(
        validation_data['wine_class'], y_validation_pred)
    y_test_pred = lasso_decision_tree_model.predict(
        test_data[lasso_selected_feature])
    lasso_decision_test_accuracy = accuracy_score(
        test_data['wine_class'], y_test_pred)
    print("Lasso DecisionTree Validation Accuracy = ",
          decision_validation_accuracy)
    print("Lasso DecisionTree Test Accuracy = ", lasso_decision_test_accuracy)


def main():
    white_wine = pd.read_csv('winequality-white.csv', sep=';')
    original_df = pd.read_csv('testfile.csv', sep=';')
    outcome_df = pd.read_csv('expected_testfile_output.csv', sep=';')
    not_null_outcome_df = pd.read_csv('not_null_expected_outcome.csv', sep=';')
    white_wine = data_processing.check_null(white_wine)
    white_wine = data_processing.split_quality(white_wine)
    white_wine = data_processing.rename_columns(white_wine)
    normal_df = data_processing.normalize(white_wine)
    correlation_graph(white_wine)
    features_subplot(white_wine)
    target_count(white_wine)
    balanced_target_count(white_wine)
    machine_learning(white_wine, normal_df)
    testfile.test_check_null(not_null_outcome_df, original_df)
    testfile.test_split_quality(outcome_df, not_null_outcome_df)


if __name__ == '__main__':
    main()
