# Personality recognition

## Data

myPersonality is a sample of personality scores and Facebook profile data collected by David Stillwell and Michal Kosinski through a Facebook application that implements the NEO-PI-R (five factor personality test, Costa & McCrae) and other psychological tests. This dataset is a subset containing 250 users and about 9900 status updates. It includes Facebook statuses in raw text, gold standard personality labels both as scores and as classes derived froma median split of the scores. It also includes several social network measures such as: network size, betweenness centrality, density, brokerage, transitivity.

The status updates in myPersonality have been anonymized manually, with each proper name replaced by a fixed string (\*PROPNAME\*), though famous named entities such as Chopin or New York have not been replaced.

~~The data has been split into 80% train and 20% test. You will receive the training subset. The class distributions of the training data are shown in train_splits.png; the test data is also imbalanced but may have different distributions.~~

## Usage

**!! Please install Python on your computer first**. See [How To Install Python on Windows, macOS, and Linux
](https://kinsta.com/knowledgebase/install-python/).

`personality_classifier.py` will load the data, run the feature extractor with the supplied configuration file `config.ini`, and train a model to predict each personality trait's binary label  with the extracted features automatically.

`config.ini` includes raw feature names in our data. You can specific any features by column names appeared in `csv` files(See `mypersonality_train.csv` for more details). For example, in default `config.ini` file we use `sEXT`, `sNEU`, `sAGR`, `sCON` and `sOPN` as features to build our classifier. For those with programming experience, you can create higher-order features based on original data or modify model parameters to get better performance.

You can also run `personality_classifier.py` with the `--load` option to have it load the binary model files stored in the train mode and run them on new data. Execute personality_classifier.py without arguments for usage.

### Examples

quickstart: `python personality_classifier.py`

training on custom dataset with custom configuration file: `python personality_classifier.py -d path_to_your_data -c path_to_your_configuration`

load well trained model: `python personality_classifier.py -l`

see more information about command arguments: `python personality_classifier.py -h`

## Submission

Each team will have three tries to submit an experiment configuration including five pickled models (one for each trait), and a conf file; the configuration will be run on the unseen test data. You can also submit a modified personality_classifier.py if you would like to change the machine learning parameters.

## Evaluation

Evaluation just looks at percent accuracy. More important than accuracy is the thoughtful use of motivated and interesting features.

## Individual component

Each team member should also submit a discussion of the features used and discarded, and results, due via email (um.andlab@gmail.com) next week Wednesday(March 17).

## Important dates for the project

**5/17**: 10-minute presentation of your proposal in class (email me slides or link to slides before class)

**5/24**: Paper due 11:59pm
