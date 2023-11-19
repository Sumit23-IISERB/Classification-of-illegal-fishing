from classification import Classification
from predict import ModelPredictor
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def main():

    list=['dt','rf','ab']
    for i in list:
        classifier = Classification(clf_opt=i, no_of_selected_features=5)
        classifier.classification()
    
        

if __name__ == "__main__":
    main()
