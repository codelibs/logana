import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Protocol, TypeVar

import tensorflow as tf
from tensorflow_serving.apis import input_pb2

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_bytes_feature(tokens: List[bytes]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=tokens))


def get_str_feature(tokens: List[str]) -> tf.train.Feature:
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[s.encode("utf-8") for s in tokens])
    )


def get_float_feature(tokens: List[float]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.FloatList(value=tokens))


def get_int64_feature(tokens: List[int]) -> tf.train.Feature:
    return tf.train.Feature(bytes_list=tf.train.Int64List(value=tokens))


class Analyzer(Protocol):
    def __call__(self, __origin: T) -> T:
        return __origin


class Featurizer(Protocol):
    def __call__(self, __origin: T) -> tf.train.Feature:
        return tf.train.Feature()


@dataclasses.dataclass
class TfRecordField:
    name: str
    proerty_name: str
    analyzer: Analyzer
    featurizer: Featurizer
    dictionary: Optional[str] = None
    primary: bool = True


@dataclasses.dataclass
class TfRecordConfig:
    example_fields: List[TfRecordField]
    context_fields: List[TfRecordField]
    train_path: str = ""
    eval_path: str = ""
    split_ratio: float = 0.1
    min_title_df: int = 3
    action_name: str = "boolean.clicked"
    relevance_level: int = 5
    min_total: int = 0
    min_action: int = -1
    to_ndjson: bool = False


def create_elwc(features: Dict[str, tf.train.Feature]) -> Any:
    elwc: Any = input_pb2.ExampleListWithContext()
    context_feature = tf.train.Example(features=tf.train.Features(feature=features))
    elwc.context.CopyFrom(context_feature)
    return elwc


def add_example(
    elwc: Any, relevance: int, features: Dict[str, tf.train.Feature]
) -> None:
    if relevance is not None:
        features["relevance"] = tf.train.Feature(
            int64_list=tf.train.Int64List(value=[relevance])
        )
    example_feature = tf.train.Example(features=tf.train.Features(feature=features))
    elwc.examples.add().CopyFrom(example_feature)
