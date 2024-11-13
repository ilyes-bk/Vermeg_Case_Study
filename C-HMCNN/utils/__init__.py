import os

datasets = {
    'originaldefects_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/originaldefects_train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/originaldefects_test.arff'
    ),
   
     'finaldefects_others': (
        False,
        os.environ['DATA_FOLDER'] + '/HMC_data/others/finaldefects_train.arff',
        os.environ['DATA_FOLDER'] + '/HMC_data/others/finaldefects_test.arff'
    ),
    
}
