"""
This is a boilerplate pipeline 'preprocessing'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node
from .nodes import clean_data, feature_engineering, normalize_features

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=clean_data,
                inputs="raw_transactions",
                outputs="clean_transactions",
                name="clean_data_node",
            ),
            node(
                func=feature_engineering,
                inputs="clean_transactions",
                outputs="engineered_features",
                name="feature_engineering_node",
            ),
            node(
                func=normalize_features,
                inputs="engineered_features",
                outputs="normalized_features",
                name="normalize_features_node",
            ),
        ]
    )
