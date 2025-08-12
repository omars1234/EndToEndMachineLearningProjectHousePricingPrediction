from src.pipeline.data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.data_cleaning import DataCleaningPipeline
from src.pipeline.data_transformation import DataTransformationPipeline
from src.pipeline.data_training import DataTrainingPipeline
from src.pipeline.model_evaluation import ModelEvaluationPipeline


def test_DataIngestionTrainingPipeline():
        assert DataIngestionTrainingPipeline()

def test_DataCleaningPipeline():
        assert DataCleaningPipeline()        


