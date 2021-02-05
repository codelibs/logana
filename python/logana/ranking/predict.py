import gzip
import json
import logging

import tensorflow as tf

from logana import ranking
from logana.common import NumpyJsonEncoder
from logana.ranking.reader import NdjsonReader

logger = logging.getLogger(__name__)


def save_predictions(
    config: ranking.TfRecordConfig,
    saved_model_path: str,
    ndjson_path: str,
    output_path: str,
) -> None:
    reader: NdjsonReader = NdjsonReader(ndjson_path)
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
                        {
                            "keyword": example.get("keyword"),
                            "title": context.get("title"),
                            "score": score,
                            "relevance": context.get("relevance"),
                        },
                        ensure_ascii=False,
                        cls=NumpyJsonEncoder,
                    )
                )
                writer.write("\n")
                context_count += 1
        example_count += 1
    logger.info(f"Predict {example_count} examples : {context_count} contexts")
