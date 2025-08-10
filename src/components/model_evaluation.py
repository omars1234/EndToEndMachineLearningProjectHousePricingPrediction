from src.config.configuration import ModelEvaluationConfig
from src.components.data_training import DataTraining
import joblib
import os
from constants.main import *
import numpy as np

from sklearn.metrics import r2_score,mean_squared_error
from sklearn.model_selection import cross_val_score
import json

class DataEvaluation:
    def __init__(self,model_evaluation_config:ModelEvaluationConfig=ModelEvaluationConfig()):
        """
        :param data_ingestion_config: configuration for data ingestion
        """
        self.model_evaluation_config = model_evaluation_config
 
        
    def evaluation(self):
        X_train, X_val, y_train, y_val=DataTraining().trainig()
        model=joblib.load(self.model_evaluation_config.model_dir)
      
        neg_rmse_scores = cross_val_score(model, X_train, y_train,cv=self.model_evaluation_config.cv,scoring='neg_root_mean_squared_error')

        rmse_scores = -neg_rmse_scores
        rmse_scores= np.round(rmse_scores, 2)
        mean_rmse_scores= round(rmse_scores.mean(), 2)
        std_rmse_scores= round(rmse_scores.std(), 2)
        Standard_Error_SE=std_rmse_scores/np.sqrt(self.model_evaluation_config.cv)
        ci_lower = mean_rmse_scores - 2* Standard_Error_SE
        ci_upper = mean_rmse_scores + 2* Standard_Error_SE  
        ci=[ci_lower, ci_upper]

        CV_results={"rmse_CV_scores":rmse_scores.tolist(),"mean_rmse_scores":mean_rmse_scores,"std_rmse_scores":std_rmse_scores,
                    "Standard_Error_SE":Standard_Error_SE,"ci":ci}

        prediction=model.predict(X_val)

        MSE=mean_squared_error(y_val,prediction)
        RMSE=np.sqrt(MSE)
        r2=r2_score(y_val,prediction)

        scores={"RMSE":RMSE,"MSE":MSE,"r2":r2}

        all_results_dict=[CV_results,scores]
        with open(os.path.join(self.model_evaluation_config.evaluation_file_dir,EVALUATION_FILE_DIR_NAME), "w") as f:
            json.dump(all_results_dict,f,indent=4)

