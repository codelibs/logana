import logging

import fugashi
import tensorflow as tf
from absl import flags
from loganary.common import japanese_tokenizer, setup_logging, setup_seed
from loganary.fess.reader import SearchLogReader
from loganary.ranking.data import TfRecordConfig, TfRecordField, get_str_feature

flags.DEFINE_string("searchlog_path", None, "Path of search log files.")
flags.DEFINE_string("train_path", None, "Path of .tfrecords file for training.")
flags.DEFINE_string("eval_path", None, "Path of .tfrecords file for evaluation.")
flags.DEFINE_string("keyword_path", None, "Path of vocabulary file for keyword field.")
flags.DEFINE_string("title_path", None, "Path of vocabulary file for title field.")
flags.DEFINE_float("split_ratio", 0.1, "Split raito for evaluation dataset.")
flags.DEFINE_bool("to_ndjson", True, "Store data as ndjson files.")
flags.DEFINE_bool("verbose", False, "Set a logging level as debug.")

FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(_):
    setup_seed()
    setup_logging(FLAGS.verbose)

    tagger: fugashi.Tagger = fugashi.Tagger("-Owakati")
    config: TfRecordConfig = TfRecordConfig(
        train_path=FLAGS.train_path,
        eval_path=FLAGS.eval_path,
        example_fields=[
            TfRecordField(
                name="keyword",
                proerty_name="keyword.search_word",
                analyzer=lambda s: japanese_tokenizer(tagger, s),
                featurizer=get_str_feature,
                dictionary=FLAGS.keyword_path,
            )
        ],
        context_fields=[
            TfRecordField(
                name="title",
                proerty_name="keyword.digest",
                analyzer=lambda s: japanese_tokenizer(tagger, s),
                featurizer=get_str_feature,
                dictionary=FLAGS.title_path,
            )
        ],
        to_ndjson=FLAGS.to_ndjson,
    )
    reader: SearchLogReader = SearchLogReader(FLAGS.searchlog_path)
    reader.to_tfrecords(config)


if __name__ == "__main__":
    flags.mark_flag_as_required("searchlog_path")
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("eval_path")
    flags.mark_flag_as_required("keyword_path")
    flags.mark_flag_as_required("title_path")

    tf.compat.v1.app.run()
