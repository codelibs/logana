import logging
from typing import Any, Dict

import fugashi
import tensorflow as tf
from absl import flags
from loganary.ranking.common import japanese_tokenizer, setup_logging, setup_seed
from loganary.ranking.data import TfRecordConfig, TfRecordField, get_str_feature
from loganary.ranking.predict import save_predictions

flags.DEFINE_string("model_path", None, "Path of trained model file.")
flags.DEFINE_string("ndjson_path", None, "Path of data file.")
flags.DEFINE_string("output_path", None, "Path of prediction file.")
flags.DEFINE_string("keyword_path", None, "Path of vocabulary file for keyword field.")
flags.DEFINE_string("title_path", None, "Path of vocabulary file for title field.")
flags.DEFINE_bool("verbose", False, "Set a logging level as debug.")

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(_):
    setup_seed()
    setup_logging(FLAGS.verbose)

    tagger: fugashi.Tagger = fugashi.Tagger("-Owakati")
    config: TfRecordConfig = TfRecordConfig(
        example_fields=[
            TfRecordField(
                name="keyword",
                proerty_name="keyword.keyword",
                analyzer=lambda s: japanese_tokenizer(tagger, s),
                featurizer=get_str_feature,
                dictionary=FLAGS.keyword_path,
            )
        ],
        context_fields=[
            TfRecordField(
                name="title",
                proerty_name="keyword.title",
                analyzer=lambda s: japanese_tokenizer(tagger, s),
                featurizer=get_str_feature,
                dictionary=FLAGS.title_path,
            )
        ],
    )

    def convert_func(data: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if "contexts" in data:
            for i, doc in enumerate(data["contexts"]):
                relevance: int = doc.get("relevance")
                results[f"doc_{i+1}"] = {
                    "id": f"{i+1}",
                    "keyword": {
                        "title": doc.get("title"),
                    },
                    "boolean": {
                        "clicked": relevance > 0,
                    },
                    "integer": {
                        "relevance": relevance,
                    },
                }

        return {
            "request": {
                "id": {
                    "query": data.get("_id"),
                },
                "conditions": {
                    "keyword": {
                        "keyword": data.get("keyword"),
                    },
                },
            },
            "response": {
                "results": results,
            },
            "@timestamp": "now",
        }

    def dump_func(
        example: Dict[str, Any], context: Dict[str, Any], score: float
    ) -> Dict[str, Any]:
        return {
            "keyword": example.get("keyword"),
            "title": context.get("title"),
            "score": score,
            "relevance": context.get("relevance"),
        }

    save_predictions(
        config,
        FLAGS.model_path,
        FLAGS.ndjson_path,
        FLAGS.output_path,
        convert_func,
        dump_func,
    )


if __name__ == "__main__":
    flags.mark_flag_as_required("model_path")
    flags.mark_flag_as_required("ndjson_path")
    flags.mark_flag_as_required("output_path")

    tf.compat.v1.app.run()
