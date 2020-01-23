#Thanks to Analytic Vidhya
#See: https://medium.com/analytics-vidhya/tracking-deep-learning-experiments-using-keras-mlflow-and-mongodb-732fc771266c
import json
import mlflow
import sys
sys.path.append('model/')
sys.path.append('data/')
#from train import train
#from data import get_data

def main():
    with open('training_conf.json') as f:
        data = json.load(f)
        print(data)
    
    keras_model = model(data['opt'])
    trainX, trainY, testX, testY = get_data()
    
    with mlflow.start_run(run_name=data['name']):    
        trained_model=train(trainX, trainY, keras_model)
        scores=keras_model.evaluate(testX, testY,verbose=1)
        mlflow.log_param("alpha", 0.001)
        mlflow.log_param("epochs", data['epochs'])
        mlflow.log_param("optimizer", data['opt'])
        mlflow.log_param("batch_size", data['batch_size'])
        mlflow.log_param("eval_loss", scores[0])
        mlflow.log_param("val_acc", scores[1])
        #mlflow.log_param("eval_precision", scores[2])
        #mlflow.log_param("eval_recall", scores[3])
        mlflow.log_param(key="accuracy", value=scores[4], step=dataset_count)

if __name__ == "__main__":
    main()        