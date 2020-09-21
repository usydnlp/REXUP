from schema import Schema, And, Or, Use, Optional
import numpy as np


instance_schema = Schema({
    "answer": And(str, len),
    "group": And(str, len),
    "imageId": {
        "group": And(str, len),
        "id": And(str, len),
        "idx": And(int, lambda x: x >= 0)
    },
    "index": And(int, lambda x: x >= 0),
    "objectsNum": And(int, lambda x: x >= 0),
    "question": And(list, len),
    "questionId": And(str, len),
    "questionStr": And(str, len),
    "sceneObjectsNum": And(int, lambda x: x >= 0),
    "tier": And(str, len),
    "type": And(str, len)
})

vectorised_data_schema = Schema({
    "answers": np.array,
    "imageIds": [{
        "group": And(str, len),
        "id": And(str, len),
        "idx": And(int, lambda x: x >= 0)
    }],
    "indices": [
        Schema(int)
    ],
    "instances": [
        instance_schema
    ],
    Optional("objectsNums"): [
        Schema(int)
    ],
    "questionLengths": np.array,
    "questions": np.array,
    Optional("sceneObjectsNums"): [
        Schema(int)
    ] # TODO Include optional info for when config.dataset == "VQA"
})

tier_schema = Schema({
    "data": [
        vectorised_data_schema
    ],
    "images": {
        str: {
            "imagesFilename": str, # config.imagesFile(tier)
            "imgsInfoFilename": str, # config.imgsInfoFile(tier)
            "imgsSceneGraph": str # config.sceneGraphsFile(tier)
        }
    },
    "train": bool
})

dataset_schema = Schema({
    "evalTrain": tier_schema,
    "test": tier_schema,
    "train": Or(None, tier_schema),
    "val": tier_schema,
})

data_schema = Schema({
    "main": Or(None, dataset_schema),
    "extra": Or(None, dataset_schema)
})

separate_embeddings_schema = Schema({
    'a': Or(None, np.array),
    'q': np.array,
    'scene': Or(None, np.array)
})

shared_embeddings_schema = Schema({
    'qa': np.array,
    'oeAnswers': np.array,
    'scene': Or(None, np.array)
})

if __name__ == '__main__':
    test_data = {
        "main": {
            "evalTrain": None,
            "test": None,
            "train": None,
            "val": None,
        },
        "extra": None
    }
    data_schema.validate(test_data)