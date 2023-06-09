"""
This is a boilerplate pipeline 'modeling'
generated using Kedro 0.18.3
"""

from kedro.pipeline import Pipeline, node, pipeline

from .nodes import dropUnusedCollumns, repairData


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=dropUnusedCollumns,
                inputs="heart_failure_raw",
                outputs="heart_failure_prepared1",
                name="drop_columns_node",
            ),
            node(
                func=repairData,
                inputs="heart_failure_prepared1",
                outputs="heart_failure_prepared",
                name="repair_data_node",
            )
        ]
    )
