import gzip
import json
import logging
from typing import Callable

import tensorflow as tf
from loganary import ranking
from loganary.ranking.common import NumpyJsonEncoder
from loganary.ranking.reader import NdjsonReader

logger = logging.getLogger(__name__)


def save_predictions(
    config: ranking.TfRecordConfig,
    saved_model_path: str,
    ndjson_path: str,
    output_path: str,
    convert_func: Callable,
    dump_func: Callable,
) -> None:
    reader: NdjsonReader = NdjsonReader(ndjson_path, convert_func)
    ranker = tf.saved_model.load(saved_model_path)
    example_count: int = 0
    context_count: int = 0
    with gzip.open(output_path, mode="wt", encoding="utf-8") as writer:
        for example, contexts, elwc in reader.read_with_tfrecords(config):
            predictions = ranker.signatures["serving_default"](
                tf.constant([elwc.SerializeToString()])
            )["outputs"]

            for context, score in zip(contexts, predictions[0].numpy()):
                writer.write(
                    json.dumps(
                        dump_func(example, context, score),
                        ensure_ascii=False,
                        cls=NumpyJsonEncoder,
                    )
                )
                writer.write("\n")
                context_count += 1
        example_count += 1
    logger.info(f"Predict {example_count} examples : {context_count} contexts")
