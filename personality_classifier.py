# label a status update for personality
import os
import pickle
import argparse
import pandas as pd
from sklearn import svm, metrics
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict, train_test_split

output_path = "./model"
traits = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]


def load_conf_file(conffile):
    conf = [line.strip() for line in open(conffile)]
    return conf


def handle_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--datafile",
        default="mypersonality_train.csv",
        help="file containing data for training or testing",
        dest="datafile",
    )
    parser.add_argument(
        "-c",
        "--conffile",
        default="config.ini",
        help="file containing list of features to be extracted",
        dest="conffile",
    )
    parser.add_argument(
        "-l",
        "--load",
        action="store_true",
        help="include to load models instead of training new",
    )
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = handle_args()
    print(args)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    df = pd.read_csv(args["datafile"], encoding="latin-1")

    # read configure
    conf = load_conf_file(args["conffile"])

    # get features and labels and split to train&&test data
    X_train, X_test, y_train, y_test = train_test_split(
        df[conf], df[traits], test_size=0.2, random_state=0
    )

    if not args["load"]:
        # train new models, evaluate, store
        for trait in traits:
            pipe = make_pipeline(svm.SVC(random_state=0))
            clf = pipe.fit(X_train, y_train[trait])
            predicted = cross_val_predict(clf, X_train, y_train[trait], cv=10)
            # print("%s train acc: %.2f%" % (trait, )
            print(
                "{} train acc: {:.2f}%".format(
                    trait, metrics.accuracy_score(y_train[trait], predicted) * 100
                )
            )
            print(
                "{} test acc: {:.2f}%".format(
                    trait, clf.score(X_test, y_test[trait]) * 100
                )
            )
            with open(os.path.join(output_path, "{}.pkl".format(trait)), "wb") as f:
                pickle.dump(clf, f)
    else:
        # load exist models
        for trait in traits:
            with open(os.path.join(output_path, "{}.pkl".format(trait)), "rb") as f:
                clf = pickle.load(f)
            print(
                "{} test acc: {:.2f}%".format(
                    trait, clf.score(X_test, y_test[trait]) * 100
                )
            )
