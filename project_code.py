import matplotlib.pyplot as plt
import numpy as np
import os
import pandas
from sklearn import model_selection, svm, neighbors, tree, neural_network, preprocessing, ensemble, impute

CV = 5


# Health Dataset: https://www.kaggle.com/datasets/rishidamarla/heart-disease-prediction
# Apple Dataset: https://www.kaggle.com/datasets/nelgiriyewithana/apple-quality?resource=download

def dtree(data, label, filename):
    # Initialize Parameters
    max_depth = range(1, 10)
    ccp_alpha = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # Start with learning curve
    sizes, training_scores, testing_scores = model_selection.learning_curve(tree.DecisionTreeClassifier(), data, label,
                                                                            cv=CV, scoring='accuracy')

    learn_plotter(sizes, training_scores, testing_scores, 'Learning Curve for Decision Tree', "Training Set Size",
                  "Accuracy Score", f'{filename}_dtree_learning_curve')

    # First Validation curve
    train_score, test_score = model_selection.validation_curve(tree.DecisionTreeClassifier(), data, label,
                                                               param_name='max_depth',
                                                               param_range=max_depth,
                                                               cv=CV, scoring='accuracy')

    valid_plotter(train_score, test_score, "Validation Curve with Decision Tree for Max Tree Depth",
                  'Different Max Tree Depth sizes',
                  'Accuracy', max_depth, f'{filename}_dtree_max_tree_depth')

    # Second Validation Curve
    train_score_2, test_score_2 = model_selection.validation_curve(tree.DecisionTreeClassifier(), data, label,
                                                                   param_name='ccp_alpha',
                                                                   param_range=ccp_alpha, cv=CV,
                                                                   scoring='accuracy')

    valid_plotter(train_score_2, test_score_2, "Validation Curve with Decision Tree for CCP Alpha",
                  'Different Pruning Settings',
                  'Accuracy', ccp_alpha, f'{filename}_dtree_pruning')


def neural(data, label, filename):
    # Initialize Parameters
    h_layer = [(583, 10), (583, 20), (583, 30), (583, 40), (583, 50)]
    h_layer_plot = [583 * 10, 583 * 20, 583 * 30, 583 * 40, 583 * 50]
    alpha = [0.01, 0.001, 0.0001]

    data_train, data_test, label_train, label_test = model_selection.train_test_split(data, label, test_size=0.2)
    mlp = neural_network.MLPClassifier(solver='adam', max_iter=400)
    mlp.fit(data_train, label_train)

    plt.figure()
    plt.plot(mlp.loss_curve_, label='Training')
    plt.xlabel('Epochs')
    plt.ylabel("Error")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}.jpg')
    plt.close()

    # Start with learning curve
    sizes, training_scores, testing_scores = model_selection.learning_curve(
        neural_network.MLPClassifier(solver='adam', early_stopping=True), data, label, cv=CV, scoring='accuracy')

    learn_plotter(sizes, training_scores, testing_scores, 'Learning Curve for Neural Network', "Dataset Size",
                  "Accuracy Score", f'{filename}_nn_learning_curve')

    # First Validation Curve
    train_score, test_score = model_selection.validation_curve(
        neural_network.MLPClassifier(solver='adam', early_stopping=True), data, label,
        param_name='hidden_layer_sizes',
        param_range=h_layer, cv=CV,
        scoring='accuracy')

    valid_plotter(train_score, test_score, "Validation Curve with Neural Network for Hidden Layer",
                  'Number of Neural Nodes',
                  'Accuracy', h_layer_plot, f'{filename}_nn_hidden_layer')

    # Second Validation Curve
    train_score_3, test_score_3 = model_selection.validation_curve(
        neural_network.MLPClassifier(solver='adam', early_stopping=True), data, label,
        param_name='alpha',
        param_range=alpha, cv=CV,
        scoring='accuracy')

    valid_plotter(train_score_3, test_score_3, "Validation Curve with Neural Network for Alpha",
                  'Different Alpha Settings',
                  'Accuracy', alpha, f'{filename}_nn_alpha')


def b_dtree(data, label, filename):
    # Initialize Parameters
    estimators = range(50, 200, 25)
    l_rate = [0.1, 0.01, 0.001]

    t = tree.DecisionTreeClassifier(max_depth=9, min_samples_leaf=1)

    # Start with learning curve
    sizes, training_scores, testing_scores = model_selection.learning_curve(
        ensemble.AdaBoostClassifier(t, algorithm='SAMME'), data, label,
        cv=CV, scoring='accuracy')

    learn_plotter(sizes, training_scores, testing_scores, 'Learning Curve for Boosted Decision Tree',
                  "Training Set Size",
                  "Accuracy Score", f'{filename}_b_dtree_learning_curve')

    # First Validation curve
    train_score, test_score = model_selection.validation_curve(ensemble.AdaBoostClassifier(t, algorithm='SAMME'), data,
                                                               label,
                                                               param_name='n_estimators',
                                                               param_range=estimators,
                                                               cv=CV, scoring='accuracy')

    valid_plotter(train_score, test_score, "Validation Curve with Boosted Decision Tree for Estimators",
                  'Different Estimators',
                  'Accuracy', estimators, f'{filename}_b_dtree_n_estimators')

    # Second Validation Curve
    train_score_2, test_score_2 = model_selection.validation_curve(ensemble.AdaBoostClassifier(t, algorithm='SAMME'),
                                                                   data, label,
                                                                   param_name='learning_rate',
                                                                   param_range=l_rate, cv=CV,
                                                                   scoring='accuracy')

    valid_plotter(train_score_2, test_score_2, "Validation Curve with Boosted Decision Tree for Learning Rate",
                  'Different Learning Rate Settings',
                  'Accuracy', l_rate, f'{filename}_b_dtree_l_rate')


def SvM(data, label, filename):
    # Initializing Parameters to follow
    kernel = ['poly', 'rbf', 'sigmoid']
    c_param = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]

    # Start with learning curve
    sizes, training_scores, testing_scores = model_selection.learning_curve(svm.SVC(C=5), data, label, cv=CV,
                                                                            scoring='accuracy')

    learn_plotter(sizes, training_scores, testing_scores, 'Learning Curve for SVM', "Training Set Size",
                  "Accuracy Score", f'{filename}_svm_learning_curve')

    # First Validation curve
    train_score, test_score = model_selection.validation_curve(svm.SVC(C=5), data, label, param_name='kernel',
                                                               param_range=kernel, cv=CV, scoring='accuracy')

    valid_plotter(train_score, test_score, "Validation Curve with SVM for Kernel", 'Different Kernel Types',
                  'Accuracy', kernel, f'{filename}_svm_kernel')

    # Second Validation Curve
    train_score_2, test_score_2 = model_selection.validation_curve(svm.SVC(), data, label, param_name='C',
                                                                   param_range=c_param, cv=CV, scoring='accuracy')

    valid_plotter(train_score_2, test_score_2, "Validation Curve with SVM for C Parameter", 'Different C Settings',
                  'Accuracy', c_param, f'{filename}_svm_c_param')


def knn(data, label, filename):
    # Initialize Parameters
    leaf_size = range(10, 30, 5)
    k = range(1, 10)

    # Start with learning curve
    sizes, training_scores, testing_scores = model_selection.learning_curve(neighbors.KNeighborsClassifier(), data,
                                                                            label, cv=CV, scoring='accuracy')

    learn_plotter(sizes, training_scores, testing_scores, 'Learning Curve for K-Neighbors', "Training Set Size",
                  "Accuracy Score", f'{filename}_knn_learning_curve')

    # First Validation curve
    train_score, test_score = model_selection.validation_curve(neighbors.KNeighborsClassifier(), data, label,
                                                               param_name='leaf_size',
                                                               param_range=leaf_size,
                                                               cv=CV, scoring='accuracy')

    valid_plotter(train_score, test_score, "Validation Curve with K-Neighbors for Leaf Sizes",
                  'Different Leaf Sizes',
                  'Accuracy', leaf_size, f'{filename}_knn_leaf_size')

    # Second Validation Curve
    train_score_2, test_score_2 = model_selection.validation_curve(neighbors.KNeighborsClassifier(), data, label,
                                                                   param_name='n_neighbors',
                                                                   param_range=k, cv=CV,
                                                                   scoring='accuracy')

    valid_plotter(train_score_2, test_score_2, "Validation Curve with K-Neighbors for K Parameter",
                  'Different K Settings',
                  'Accuracy', k, f'{filename}_knn_k')


def learn_plotter(sizes, train, test, Title, xlabel, ylabel, filename):
    # Mean and Standard Deviation of training scores
    mean_training = np.mean(train, axis=1)

    # Mean and Standard Deviation of testing scores
    mean_testing = np.mean(test, axis=1)

    # dotted blue line is for training scores and green line is for cross-validation score
    plt.plot(sizes, mean_training, '--', color="b", label="Training score")
    plt.plot(sizes, mean_testing, color="g", label="Cross-validation score")

    # Drawing plot
    plt.title(Title)
    plt.xlabel(xlabel), plt.ylabel(ylabel), plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}.jpg')
    plt.close()


def valid_plotter(train, test, Title, xlabel, ylabel, parameter, filename):
    # Calculating mean of training score
    mean_train_score = np.mean(train, axis=1)

    # Calculating mean of testing score
    mean_test_score = np.mean(test, axis=1)

    # Plot mean accuracy scores for training and testing scores
    plt.plot(parameter, mean_train_score,
             label="Training Score", color='b')
    plt.plot(parameter, mean_test_score,
             label="Cross Validation Score", color='g')

    # Creating the plot
    plt.title(Title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.savefig(f'outputs/{filename}.jpg')
    plt.close()


def pre_process_health(heart):
    """
    This function preprocesses the data by extracting specific data from the heart dataset as to reduce
    complexity
    :param heart: DataFrame object containing data from csv file
    :return:
        data: containing independent variables that the model will associate with classes
        label: contains classes regarding the data
    """
    data = heart.iloc[:, : - 2].values
    sex_encoder = preprocessing.LabelEncoder()
    data[:, 1] = sex_encoder.fit_transform(data[:, 1])
    label = np.array([1 if x == 2 else 0 for x in heart.iloc[:, -1].values])

    return data, label


def pre_process_apple(apples):
    data = apples.iloc[:, 1:8].values
    labels = apples.iloc[:, -1].values
    apple_encoder = preprocessing.LabelEncoder()
    labels = apple_encoder.fit_transform(labels)

    return data, labels


if __name__ == '__main__':
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    health_data, health_label = pre_process_health(pandas.read_csv('classification_dataset/health.csv'))
    apple_data, apple_label = pre_process_apple(pandas.read_csv('classification_dataset/apple_quality.csv'))

    SvM(apple_data, apple_label, 'apple')
    dtree(apple_data, apple_label, 'apple')
    knn(apple_data, apple_label, 'apple')
    b_dtree(apple_data, apple_label, 'apple')

    SvM(health_data, health_label, 'health')
    dtree(health_data, health_label, 'health')
    knn(health_data, health_label, 'health')
    b_dtree(health_data, health_label, 'health')

    neural(apple_data, apple_label, 'apple')
    neural(health_data, health_label, 'health')
