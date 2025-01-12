"""
This is a boilerplate pipeline 'models'
generated using Kedro 0.19.10
"""

from kedro.pipeline import Pipeline, node
from .nodes import train_isolation_forest, save_results

def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=train_isolation_forest,
                inputs=dict(
                    data="normalized_features",
                    features="params:model_features",
                    model_params="params:isolation_forest_params",
                ),
                outputs=["data_with_anomalies", "isolation_forest_model"],
                name="train_isolation_forest_node",
            ),
            node(
                func=save_results,
                inputs=dict(
                    data="data_with_anomalies",
                    model="isolation_forest_model",
                    result_path="params:result_path",
                    model_path="params:model_path",
                ),
                outputs=None,
                name="save_results_node",
            ),
        ]
    )
