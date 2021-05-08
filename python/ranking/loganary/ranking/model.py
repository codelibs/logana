import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import tensorflow as tf
import tensorflow_ranking as tfr
from loganary.ranking.common import get_ndcg_metric
from tensorflow.python.feature_column.feature_column_v2 import FeatureColumn
from tensorflow.python.ops.init_ops import Initializer

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class RankingModelField:
    name: str
    column_type: str = "numeric"
    default_value: Optional[int] = None

    def get_column(self) -> FeatureColumn:
        if self.column_type == "int8":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.int8, default_value=self.default_value
            )
        if self.column_type == "int16":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.int16, default_value=self.default_value
            )
        if self.column_type == "int32":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.int32, default_value=self.default_value
            )
        if self.column_type == "int64" or self.column_type == "numeric":
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
        if self.column_type == "float64":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.float64, default_value=self.default_value
            )
        if self.column_type == "double":
            return tf.feature_column.numeric_column(
                self.name, dtype=tf.double, default_value=self.default_value
            )
        raise ValueError(f"Unknown column type: {self.column_type}")


@dataclasses.dataclass
class RankingModelEmbeddingField(RankingModelField):
    vocabulary_file: Optional[str] = None
    vocabulary_size: Optional[int] = None
    num_oov_buckets: int = 0
    dimension: Optional[int] = None
    combiner: str = "mean"
    initializer: Optional[Initializer] = None
    ckpt_to_load_from = None
    tensor_name_in_ckpt = None
    max_norm = None
    trainable: bool = True
    use_safe_embedding_lookup: bool = True

    def get_column(self) -> FeatureColumn:
        categorical_column_ = tf.feature_column.categorical_column_with_vocabulary_file(
            key=self.name,
            vocabulary_file=self.vocabulary_file,
            vocabulary_size=self.vocabulary_size,
            default_value=self.default_value,
            num_oov_buckets=self.num_oov_buckets,
        )
        return tf.feature_column.embedding_column(
            categorical_column_,
            self.dimension,
            combiner=self.combiner,
            initializer=self.initializer,
            ckpt_to_load_from=self.ckpt_to_load_from,
            tensor_name_in_ckpt=self.tensor_name_in_ckpt,
            max_norm=self.max_norm,
            trainable=self.trainable,
            use_safe_embedding_lookup=self.use_safe_embedding_lookup,
        )


def _get_default_loss_keys() -> List[str]:
    return [tfr.losses.RankingLossKey.APPROX_NDCG_LOSS]


@dataclasses.dataclass
class RankingModelConfig:
    model_path: str
    train_path: str
    eval_path: str
    context_fields: List[RankingModelField]
    example_fields: List[RankingModelField]
    label_field: RankingModelField
    num_train_steps: int = 15000
    loss_keys: List[str] = dataclasses.field(default_factory=_get_default_loss_keys)
    hidden_layer_dims: List[int] = dataclasses.field(default_factory=list)
    batch_size: int = 32
    list_size: int = 120
    learning_rate: float = 0.05
    group_size: int = 1
    dropout_rate: float = 0.5
    save_checkpoints_steps: Optional[int] = 1000
    eval_metric: Dict[str, Callable] = dataclasses.field(
        default_factory=get_ndcg_metric
    )


class RankingModel:
    def __init__(self, config: RankingModelConfig) -> None:
        self._config = config

    def train(self) -> Tuple[Any, Any]:
        tf.compat.v1.reset_default_graph()

        optimizer: Any = self._make_optimizer_fn()

        ranking_head: Any = tfr.head.create_ranking_head(
            loss_fn=self._make_loss_fn(),
            eval_metric_fns=self._config.eval_metric,
            train_op_fn=self._make_train_op_fn(optimizer),
        )

        model_fn: Callable = tfr.model.make_groupwise_ranking_fn(
            group_score_fn=self._make_score_fn(),
            transform_fn=self._make_transform_fn(),
            group_size=self._config.group_size,
            ranking_head=ranking_head,
        )

        run_config: tf.estimator.RunConfig = tf.estimator.RunConfig(
            save_checkpoints_steps=self._config.save_checkpoints_steps
        )
        self._ranker: tf.estimator.Estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=self._config.model_path,
            config=run_config,
        )

        def train_input_fn() -> Tuple[Any, Any]:
            return self._input_fn(True)

        train_spec: tf.estimator.TrainSpec = tf.estimator.TrainSpec(
            input_fn=train_input_fn, max_steps=self._config.num_train_steps
        )

        def eval_input_fn() -> Tuple[Any, Any]:
            return self._input_fn(False)

        eval_spec: tf.estimator.EvalSpec = tf.estimator.EvalSpec(
            name="eval", input_fn=eval_input_fn, throttle_secs=15
        )

        return tf.estimator.train_and_evaluate(self._ranker, train_spec, eval_spec)

    def get_ranker(self) -> tf.estimator.Estimator:
        return self._ranker

    @staticmethod
    def _get_feature_columns(
        fields: List[RankingModelField],
    ) -> Dict[str, FeatureColumn]:
        columns: Dict[str, FeatureColumn] = {}
        for field in fields:
            columns[field.name] = field.get_column()
        return columns

    def _input_fn(self, is_training: bool = True) -> Tuple[Any, Any]:
        context_feature_spec: Dict[
            str, FeatureColumn
        ] = tf.feature_column.make_parse_example_spec(
            self._get_feature_columns(self._config.context_fields).values()
        )

        example_fields: List[FeatureColumn] = list(
            self._get_feature_columns(self._config.example_fields).values()
        )
        example_fields.append(self._config.label_field.get_column())
        example_feature_spec: Dict[
            str, FeatureColumn
        ] = tf.feature_column.make_parse_example_spec(example_fields)

        dataset: Any = tfr.data.build_ranking_dataset(
            file_pattern=self._config.train_path
            if is_training
            else self._config.eval_path,
            data_format=tfr.data.ELWC,
            batch_size=self._config.batch_size,
            list_size=self._config.list_size,
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
            reader=tf.data.TFRecordDataset,
            shuffle=False,
            num_epochs=None if is_training else 1,
        )
        features: Any = tf.compat.v1.data.make_one_shot_iterator(dataset).get_next()
        label: Any = tf.squeeze(features.pop(self._config.label_field.name), axis=2)
        label = tf.cast(label, tf.float32)

        return features, label

    def _make_transform_fn(self) -> Callable:
        def _transform_fn(features, mode):
            context_features, example_features = tfr.feature.encode_listwise_features(
                features=features,
                context_feature_columns=self._get_feature_columns(
                    self._config.context_fields
                ),
                example_feature_columns=self._get_feature_columns(
                    self._config.example_fields
                ),
                mode=mode,
                scope="transform_layer",
            )

            return context_features, example_features

        return _transform_fn

    def _make_score_fn(self) -> Callable:
        def _score_fn(context_features, group_features, mode, params, _config):
            with tf.compat.v1.name_scope("input_layer"):
                context_input = [
                    tf.compat.v1.layers.flatten(context_features[name])
                    for name in sorted(
                        self._get_feature_columns(self._config.context_fields)
                    )
                ]
                group_input = [
                    tf.compat.v1.layers.flatten(group_features[name])
                    for name in sorted(
                        self._get_feature_columns(self._config.example_fields)
                    )
                ]
                input_layer = tf.concat(context_input + group_input, 1)

            is_training = mode == tf.estimator.ModeKeys.TRAIN
            cur_layer = input_layer
            cur_layer = tf.compat.v1.layers.batch_normalization(
                cur_layer, training=is_training, momentum=0.99
            )

            for i, layer_width in enumerate(
                int(d) for d in self._config.hidden_layer_dims
            ):
                cur_layer = tf.compat.v1.layers.dense(cur_layer, units=layer_width)
                cur_layer = tf.compat.v1.layers.batch_normalization(
                    cur_layer, training=is_training, momentum=0.99
                )
                cur_layer = tf.nn.relu(cur_layer)
                cur_layer = tf.compat.v1.layers.dropout(
                    inputs=cur_layer,
                    rate=self._config.dropout_rate,
                    training=is_training,
                )
            logits = tf.compat.v1.layers.dense(cur_layer, units=self._config.group_size)
            return logits

        return _score_fn

    def _make_loss_fn(self) -> Callable:
        return tfr.losses.make_loss_fn(self._config.loss_keys)

    def _make_optimizer_fn(self) -> Callable:
        return tf.compat.v1.train.AdagradOptimizer(
            learning_rate=self._config.learning_rate
        )

    def _make_train_op_fn(self, optimizer: Any) -> Callable:
        def _train_op_fn(loss):
            update_ops: Any = tf.compat.v1.get_collection(
                tf.compat.v1.GraphKeys.UPDATE_OPS
            )
            minimize_op: Any = optimizer.minimize(
                loss=loss, global_step=tf.compat.v1.train.get_global_step()
            )
            train_op: Any = tf.group([update_ops, minimize_op])
            return train_op

        return _train_op_fn

    def _make_serving_input_fn(self) -> Callable:
        context_feature_spec = tf.feature_column.make_parse_example_spec(
            self._get_feature_columns(self._config.context_fields).values()
        )
        example_feature_spec = tf.feature_column.make_parse_example_spec(
            self._get_feature_columns(self._config.example_fields).values()
        )
        return tfr.data.build_ranking_serving_input_receiver_fn(
            data_format="example_list_with_context",
            context_feature_spec=context_feature_spec,
            example_feature_spec=example_feature_spec,
        )

    def save_model(self, model_path: str = None) -> str:
        if model_path is None:
            model_path = self._config.model_path
        return self._ranker.export_saved_model(
            f"{model_path}",
            self._make_serving_input_fn(),
            checkpoint_path=f"{model_path}/model.ckpt-{self._config.num_train_steps}",
        ).decode("utf-8")
