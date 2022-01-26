import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder, StandardScaler


def file_2_df(*args, **kwargs):
    """
    Takes a path to a csv like file and converts it into a pandas dataframe.
        csv like meaning each column of data is separated by a constant delimiter such as a comma or tab

    :param args:
        file_path:

    :param kwargs:
        col_delimiter:
        col_names:

    :return:
        Pandas Dataframe with Headers
    """

    col_delimiter = kwargs.pop("col_delimiter", 't')
    """Test to check if delimiter is valid"""
    if col_delimiter == 't':
        col_delimiter = '\t'
    elif col_delimiter != ',':
        raise ValueError("Invalid separator: 't' for tab separated or ',' for comma separated")

    file_data_df = pd.read_csv(args[0], sep=col_delimiter, engine='python', header=None)  # Create a data frame from the
    # file location

    col_names = kwargs.pop("col_names", [x for x in range(file_data_df.shape[1])])
    if len(col_names) != file_data_df.shape[1]:
        raise ValueError('Number of headings does not match number of columns [Headings: %(headings)s, Columns: '
                         '%(cols)s]' % {'headings': len(col_names), 'cols': file_data_df.shape[1]})

    file_data_df.columns = col_names  # Add columns to the data, 0...N if not given.

    # file_data_df = file_data_df.sample(frac=1)
    return file_data_df


def extract_dv(*args, **kwargs):
    """
    Extracts the Dependant variable from a Dataframe. Can also shuffle the Dataframe with a desired or random seed and
        can also split the data into a train test set with a desired percentage.
        Data in the Dataframe (only training if test_split is True) is normalised and
    :param args:
        Dataframe:

    :param kwargs:
        y_idx:
        seed:
        shuffle:
        test_split:
        split_percent:

    :return:
        x_data, y_data if test_split is false.
        x_test, x_train, y_test, y_train if test_split is True.

    """
    data_df = args[0]
    if type(data_df) != pd.core.frame.DataFrame:
        raise TypeError("input data must be Pandas Dataframe")

    y_idx = kwargs.pop('y_index', data_df.shape[1] - 1)
    seed = kwargs.pop('seed', None)
    shuffle = kwargs.pop('shuffle', False)
    test_split = kwargs.pop('test_split', True)
    split_percent = kwargs.pop('split_percent', 0.25)

    col_names = data_df.columns
    dv = str(col_names[y_idx])

    if shuffle:
        # data_df = data_df.sample(frac=1)
        np.random.seed(seed)
        data_df = data_df.copy()
        for _ in range(1):
            data_df.apply(np.random.shuffle, axis=0)

    if not test_split:
        x_data = data_df.iloc[:, data_df.columns != dv]
        y_data = data_df.iloc[:, data_df.columns == dv]
        return x_data, y_data

    test_rows = int(len(data_df) // (1 / split_percent))

    # Split Test and Train
    test_df = data_df.iloc[:test_rows, :]
    train_df = data_df.iloc[test_rows:, :]

    # Split X and Y
    x_test = test_df.iloc[:, data_df.columns != dv]
    y_test = test_df.iloc[:, data_df.columns == dv]

    x_train = train_df.iloc[:, data_df.columns != dv]
    y_train = train_df.iloc[:, data_df.columns == dv]

    # TODO: Scaler and Label Encoder
    # Label Encode the Dependant Variable
    lb = LabelEncoder()
    y_train = np.asarray(lb.fit_transform(y_train))
    y_test = np.asarray(lb.transform(y_test))

    # Feature Scaling - Only do to training set
    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    return x_test, x_train, y_test, y_train


def train_test_data_split(*args, **kwargs):
    """
    Splits the array of data into training and test sets.

    :param args: numpy.ndarrays of same length / shape[0].

    :param kwargs:
        shuffle: If set True, the rows of the data will be shuffled.

        test_size: The percentage of data that will be removed from the arrays and used as test data. E.g. a test_size=0.25
                will split the data 75% training and 25% test. If there is an uneven split, the remainder will go to the
                training data. E.G. 101 rows of data at test_size=0.25 will leave an uneven split (training_set=75.75,
                test_set=25.25), the remainder dat is passed to the training set.

        seed: a seed can be set so that the same split can be achieved again

    :return:
    """

    """Test to check correct number of arrays have been passed"""
    if 1 > len(args) or len(args) > 2:
        raise ValueError("Either pass ONE numpy array with both Independent and Dependent Variables included OR TWO"
                         "numpy arrays where the first one is the Independent Variable and the second one is the"
                         "Dependent Variable, no less or more")

    """Test to check each array is a numpy array"""
    for arr in args:
        if type(arr) is not np.ndarray:
            raise TypeError('Array type cannot be %s' % type(arr), ', must be numpy.ndarray')

    shuffle = kwargs.pop("shuffle", True)
    test_size = kwargs.pop("test_size", True)
    seed = kwargs.pop('seed', None)
    np.random.seed(seed)

    """Test to check all parameters are valid"""
    if kwargs:
        raise ValueError('Parameter(s) %s' % str(kwargs), 'is/are invalid. Valid options are "shuffle", "test_size", '
                                                          '"seed"')

    if len(args) is 2:
        # print(len(args[1]))
        if len(args[0]) != len(args[1][0]):
            raise ValueError


class MultinomialLogisticRegression:

    def __init__(self, x_array, y_array, seed):
        self.feature_array = x_array
        self.target_array = y_array
        self.seed = seed

        self.loss_list = np.array([])

        self.num_targets = len(np.unique(self.target_array))  # k
        self.num_features = self.feature_array.shape[1]  # M

        self.weights = np.random.rand(self.num_targets, self.num_features)  # Initial Random Weights
        self.biases = np.random.rand(self.num_targets, 1)  # Initial Random Biases

        self.probabilities = None
        self.accuracy = None
        self.predictions = None
        self.lpf = None

        np.random.seed(self.seed)  # Global seed

    def fit_lpf(self, weights, biases, feature_array):
        """
        Creates the Linear Predictor function and fits it with given weights and biases

        :param weights: Weights of the coefficients
        :param biases: Added Biases
        :param feature_array: Array of Observations and features
        :return: LPF fuction of scores
        """
        self.lpf = np.array([(weights.dot(feature_array[i].reshape(-1, 1)) + biases).reshape(-1) for i in
                             range(feature_array.shape[0])])
        return self.lpf

    @staticmethod
    def get_probabilities(lpf):
        """
        Converts logit scores into probabilities using a SoftMax function

        :param lpf: A linear predictor function
        :return: Probability matrix
        """
        return np.array([np.exp(lpf[i]) / np.sum(np.exp(lpf[i])) for i in range(lpf.shape[0])])

    @staticmethod
    def get_predictions(probs):
        """
        Converts a vector of probabilities into predictions
        :param probs: An array of probabilities
        :return: argmax predictions corresponding to the input probabilities
        """
        return [np.argmax(x) for x in probs]

    @staticmethod
    def get_accuracy(targets, predictions):
        """
        Gets the accuracy when passed the predicted results and actual results
        :param targets:
        :param predictions:
        :return: Accuracy values between 0 and 1
        """
        targets = [int(np.argmax(x)) for x in targets]
        return len([x for x in list(zip(targets, predictions)) if x[0] == x[1]]) / len(targets)

    @staticmethod
    def cross_entropy(probabilities, targets):
        """
        Cross Entropy function which calculates the loss between an array of probabilities and the corresponding array
        of target values

        :param probabilities: Array of probabilities
        :param targets: Target values corresponding to the given probabilities
        :return: the Cross Entropy Loss of the two arrays
        """

        return np.sum([-np.log(probs[target])
                       for probs, target in zip(probabilities, targets)]) / probabilities.shape[0]

    def mlr(self):
        """
        Non static method to obtain all the needed values for SGD and Accuracy
        """
        self.fit_lpf(weights=self.weights,
                     biases=self.biases,
                     feature_array=self.feature_array)
        self.probabilities = self.get_probabilities(self.lpf)
        self.predictions = self.get_predictions(self.probabilities)
        self.accuracy = self.get_accuracy(targets=self.target_array,
                                          predictions=self.predictions)

    def sgd(self, lr, num_epochs):
        """

        :param lr:
        :param num_epochs:
        :return:
        """
        self.mlr()

        features = self.feature_array
        targets = self.target_array
        targets = targets.astype(int)

        for epoch in range(num_epochs):
            self.mlr()
            probabilities = self.probabilities
            cel = self.cross_entropy(probabilities=probabilities,
                                     targets=targets)

            self.loss_list = np.append(self.loss_list, cel)

            probabilities[np.arange(features.shape[0]), targets] -= 1

            grad_weight = probabilities.T.dot(features)
            grad_biases = np.sum(probabilities, axis=0).reshape(-1, 1)

            self.weights -= (lr * grad_weight)
            self.biases -= (lr * grad_biases)

            if epoch % 1000 == 0:
                print('Accuracy:', self.accuracy, 'Epoch:', epoch, "CEL:", cel)

    def predict(self, x_test, y_test):
        """
        Takes test data and predict
        :param x_test: Observations to be predicted
        :param y_test: Target array, only needed if testing accuracy
        :return: predictions and accuracy
        """
        self.fit_lpf(self.weights, self.biases, x_test)
        self.probabilities = self.get_probabilities(self.lpf)
        self.predictions = self.get_predictions(self.probabilities)
        if y_test is not None:
            self.accuracy = self.get_accuracy(targets=y_test,
                                              predictions=self.predictions)
            return self.predictions, self.accuracy
        return self.predictions, None


if __name__ == '__main__':
    file_path = "/Users/eoinmac/PycharmProjects/MachineLearning/Assignment_2/beer.txt"

    # Set headings for features
    cols = ["calorific_value", "nitrogen",
            "turbidity", "style",
            "alcohol", "sugars",
            "bitterness", "beer_id",
            "colour", "degree_of_fermentation"]

    # Set test variables
    random_seed = 1
    target_variable = 'style'
    target_idx = cols.index(target_variable)
    training_set_split_percent = 0.3
    epochs = 10_000
    learning_rate = 0.005

    # Create Data frame from the file
    data_df = file_2_df(file_path,
                        col_delimiter='t',
                        col_names=cols)

    # Extract Dependent Variable and split into training and testing sets
    x_test, x_train, y_test, y_train = extract_dv(data_df,
                                                  y_index=target_idx,
                                                  seed=random_seed,
                                                  shuffle=True,
                                                  test_split=True,
                                                  split_percent=training_set_split_percent)

    # Create a MLR class and fit the data to the function
    mlr = MultinomialLogisticRegression(x_array=x_train,
                                        y_array=y_train,
                                        seed=random_seed)

    mlr.mlr()
    mlr.sgd(learning_rate, epochs)
    mlr.mlr()
    y_pred, acc = mlr.predict(x_test, y_test)
    print(len(y_test), len(y_pred))
    print(acc)

    from sklearn.metrics import confusion_matrix

    confusion_matrix = confusion_matrix(y_test, y_pred)
    print(confusion_matrix, 'cm')

    from sklearn.metrics import classification_report

    print(classification_report(y_test, y_pred), 'report')
    print('\n')
    print('\n')
    print('\n')
    print('\n')

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    from sklearn.metrics import roc_auc_score
    from sklearn.metrics import roc_curve

    from matplotlib import pyplot as plt

    logit_roc_auc = roc_auc_score(y_test, mlr.predict(x_test, y_test))
    fpr, tpr, thresholds = roc_curve(y_test, mlr.predict_proba(x_test)[:, 1])
    plt.figure()
    plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.savefig('Log_ROC')
    plt.show()

