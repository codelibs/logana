import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_ranking as tfr
from tensorflow.python.feature_column.feature_column_v2 import FeatureColumn

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class TfRankingModelField:
    name: str
    column_type: str = "embedding"
    dictionary: Optional[str] = None
    dimension: Optional[int] = None
    default_value: Optional[int] = None

    def get_column(self) -> FeatureColumn:
        if self.column_type == "embedding":
            categorical_column_ = (
                tf.feature_column.categorical_column_with_vocabulary_file(
                    key=self.name, vocabulary_file=self.dictionary
                )
            )
            return tf.feature_column.embedding_column(
                categorical_column_, self.dimension
            )
        if self.column_type == "int32":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.int32, default_value=self.default_value
            )
        if self.column_type == "numeric" or self.column_type == "int64":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.int64, default_value=self.default_value
            )
        if self.column_type == "float16":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.float16, default_value=self.default_value
            )
        if self.column_type == "float32":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.float32, default_value=self.default_value
            )
        raise ValueError(f"Unknown column type: {self.column_type}")


def get_ndcg_metric(topn: List[int] = [10, 20, 30, 40, 50]) -> Dict[str, Callable]:
    return {
        "metric/ndcg@%d"
        % x: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=x
        )
        for x in topn
    }


@dataclasses.dataclass
class TfRankingModelConfig:
    model_path: str
    train_path: str
    eval_path: str
    example_fields: List[TfRankingModelField]
    context_fields: List[TfRankingModelField]
    label_field: TfRankingModelField
    num_train_steps: int = 15000
    hidden_layer_dims: List[int] = dataclasses.field(default_factory=list)
    batch_size: int = 32
    list_size: int = 120
    learning_rate: float = 0.05
    group_size: int = 1
    dropout_rate: float = 0.5
    eval_metric: Dict[str, Callable] = dataclasses.field(
        default_factory=get_ndcg_metric
    )


def _get_context_feature_columns(
    config: TfRankingModelConfig,
) -> Dict[str, FeatureColumn]:
    columns: Dict[str, FeatureColumn] = {}
    for field in config.context_fields:
        columns[field.name] = field.get_column()
    return columns


def _get_example_feature_columns(
    config: TfRankingModelConfig,
) -> Dict[str, FeatureColumn]:
    columns: Dict[str, FeatureColumn] = {}
    for field in config.example_fields:
        columns[field.name] = field.get_column()
    return columns


def _input_fn(config: TfRankingModelConfig, is_train: bool = True) -> Tuple[Any, Any]:
    context_feature_spec: Dict[
        str, FeatureColumn
    ] = tf.feature_column.make_parse_example_spec(
        _get_context_feature_columns(config).values()
    )

    example_columns: List[FeatureColumn] = list(
        _get_example_feature_columns(config).values()
    )
    example_columns.append(config.label_field.get_column())
    example_feature_spec: Dict[
        str, FeatureColumn
    ] = tf.feature_column.make_parse_example_spec(example_columns)

    dataset: Any = tfr.data.build_ranking_dataset(
        file_pattern=config.train_path if is_train else config.eval_path,
        data_format=tfr.data.ELWC,
        batch_size=config.batch_size,
        list_size=config.list_size,
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
        reader=tf.data.TFRecordDataset,
        shuffle=False,
        num_epochs=None if is_train else 1,
    )
    features: Any = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
    label: Any = tf.squeeze(features.pop(config.label_field.name), axis=2)
    label = tf.cast(label, tf.float32)

    return features, label


def _make_transform_fn(config: TfRankingModelConfig) -> Callable:
    def _transform_fn(features, mode):
        context_features, example_features = tfr.feature.encode_listwise_features(
            features=features,
            context_feature_columns=_get_context_feature_columns(config),
            example_feature_columns=_get_example_feature_columns(config),
            mode=mode,
            scope="transform_layer",
        )

        return context_features, example_features

    return _transform_fn


def _make_score_fn(config: TfRankingModelConfig) -> Callable:
    def _score_fn(context_features, group_features, mode, params, _config):
        with tf.compat.v1.name_scope("input_layer"):
            context_input = [
                tf.compat.v1.layers.flatten(context_features[name])
                for name in sorted(_get_context_feature_columns(config))
            ]
            group_input = [
                tf.compat.v1.layers.flatten(group_features[name])
                for name in sorted(_get_example_feature_columns(config))
            ]
            input_layer = tf.concat(context_input + group_input, 1)

        is_training = mode == tf.estimator.ModeKeys.TRAIN
        cur_layer = input_layer
        cur_layer = tf.compat.v1.layers.batch_normalization(
            cur_layer, training=is_training, momentum=0.99
        )

        for i, layer_width in enumerate(int(d) for d in config.hidden_layer_dims):
            cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training, momentum=0.99
            )
            cur_layer = tf.nn.relu(cur_layer)
            cur_layer = tf.compat.v1.layers.dropout(
                inputs=cur_layer, rate=config.dropout_rate, training=is_training
            )
        logits = tf.compat.v1.layers.dense(cur_layer, units=config.group_size)
        return logits

    return _score_fn


def run_train(config: TfRankingModelConfig) -> Tuple[Any, Any]:
    tf.compat.v1.reset_default_graph()

    loss: Any = tfr.losses.RankingLossKey.APPROX_NDCG_LOSS
    loss_fn: Any = tfr.losses.make_loss_fn(loss)

    optimizer: Any = tf.compat.v1.train.AdagradOptimizer(
        learning_rate=config.learning_rate
    )

    def _train_op_fn(loss):
        update_ops: Any = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        minimize_op: Any = optimizer.minimize(
            loss=loss, global_step=tf.compat.v1.train.get_global_step()
        )
        train_op: Any = tf.group([update_ops, minimize_op])
        return train_op

    ranking_head: Any = tfr.head.create_ranking_head(
        loss_fn=loss_fn, eval_metric_fns=config.eval_metric, train_op_fn=_train_op_fn
    )

    model_fn: Callable = tfr.model.make_groupwise_ranking_fn(
        group_score_fn=_make_score_fn(config),
        transform_fn=_make_transform_fn(config),
        group_size=config.group_size,
        ranking_head=ranking_head,
    )

    def train_input_fn() -> Tuple[Any, Any]:
        return _input_fn(config)

    def eval_input_fn() -> Tuple[Any, Any]:
        return _input_fn(config, is_train=False)

    run_config: tf.estimator.RunConfig = tf.estimator.RunConfig(
        save_checkpoints_steps=1000
    )
    ranker: tf.estimator.Estimator = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=config.model_path, config=run_config
    )

    train_spec: tf.estimator.TrainSpec = tf.estimator.TrainSpec(
        input_fn=train_input_fn, max_steps=config.num_train_steps
    )
    eval_spec: tf.estimator.EvalSpec = tf.estimator.EvalSpec(
        name="eval", input_fn=eval_input_fn, throttle_secs=15
    )

    result = tf.estimator.train_and_evaluate(ranker, train_spec, eval_spec)
    return ranker, result


def _make_serving_input_fn(config: TfRankingModelConfig) -> Callable:
    context_feature_spec = tf.feature_column.make_parse_example_spec(
        _get_context_feature_columns(config).values()
    )
    example_feature_spec = tf.feature_column.make_parse_example_spec(
        _get_example_feature_columns(config).values()
    )
    return tfr.data.build_ranking_serving_input_receiver_fn(
        data_format="example_list_with_context",
        context_feature_spec=context_feature_spec,
        example_feature_spec=example_feature_spec,
    )


def save_model(ranker: tf.estimator.Estimator, config: TfRankingModelConfig) -> str:
    return ranker.export_saved_model(
        f"{config.model_path}",
        _make_serving_input_fn(config),
        checkpoint_path=f"{config.model_path}/model.ckpt-{config.num_train_steps}",
    ).decode("utf-8")
