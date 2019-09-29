from allennlp.predictors.predictor import Predictor

inputStr = 'A dog is laying in the grass with a frisbee in mouth'

if __name__ == '__main__':
    predictor = Predictor.from_path("allennlp_models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    val = predictor.predict(sentence=inputStr)

    print(val)
    
