import dataclasses
import datetime
import gzip
import json
import logging
import os
from typing import Any, Dict

import numpy as np
import tensorflow as tf
from absl import flags
from loganary.ranking.common import NumpyJsonEncoder, setup_logging, setup_seed
from loganary.ranking.model import (
    RankingModel,
    RankingModelConfig,
    RankingModelEmbeddingField,
    RankingModelField,
)

flags.DEFINE_string("train_path", None, "Path of .tfrecords file for training.")
flags.DEFINE_string("eval_path", None, "Path of .tfrecords file for evaluation.")
flags.DEFINE_string("keyword_path", None, "Path of vocabulary file for keyword field.")
flags.DEFINE_string("title_path", None, "Path of vocabulary file for title field.")
flags.DEFINE_string("model_path", None, "Path of trained model files.")
flags.DEFINE_integer("num_train_steps", 15000, "The number of train steps.")
flags.DEFINE_list("hidden_layer_dims", ["64", "32", "16"], "Sizes for hidden layers.")
flags.DEFINE_integer(
    "keyword_embedding_dim", 20, "Dimention of an embedding for keyword field."
)
flags.DEFINE_integer(
    "title_embedding_dim", 20, "Dimention of an embedding for title field."
)
flags.DEFINE_integer("batch_size", 32, "Batch size.")
flags.DEFINE_integer("list_size", 100, "List size.")
flags.DEFINE_float("learning_rate", 0.05, "Learning rate.")
flags.DEFINE_integer("group_size", 10, "Group size.")
flags.DEFINE_float("dropout_rate", 0.8, "Dropout rate.")
flags.DEFINE_bool("verbose", False, "Set a logging level as debug.")


FLAGS = flags.FLAGS
logger = logging.getLogger(__name__)


def main(_) -> None:
    setup_seed()
    setup_logging(FLAGS.verbose)

    now_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
    model_path: str = f"{FLAGS.model_path}/{now_str}"
    config: RankingModelConfig = RankingModelConfig(
        model_path=model_path,
        train_path=FLAGS.train_path,
        eval_path=FLAGS.eval_path,
        context_fields=[
            RankingModelEmbeddingField(
                name="keyword",
                vocabulary_file=FLAGS.keyword_path,
                dimension=FLAGS.keyword_embedding_dim,
            ),
        ],
        example_fields=[
            RankingModelEmbeddingField(
                name="title",
                vocabulary_file=FLAGS.title_path,
                dimension=FLAGS.title_embedding_dim,
            ),
        ],
        label_field=RankingModelField(
            name="relevance",
            column_type="numeric",
            default_value=-1,
        ),
        num_train_steps=FLAGS.num_train_steps,
        hidden_layer_dims=FLAGS.hidden_layer_dims,
        batch_size=FLAGS.batch_size,
        list_size=FLAGS.list_size,
        learning_rate=FLAGS.learning_rate,
        group_size=FLAGS.group_size,
        dropout_rate=FLAGS.dropout_rate,
    )
    logger.info(f"Config: {config}")
    model: RankingModel = RankingModel(config)
    result = model.train()
    logger.info(f"Result: {result}")
    export_model_path: str = model.save_model()
    saved_model_path: str = f"{model_path}/saved_model"
    os.rename(export_model_path, saved_model_path)
    logger.info(f"Output Model Path: {saved_model_path}")

    with gzip.open(f"{model_path}/result.json.gz", mode="wt", encoding="utf-8") as f:
        config_dict: Dict[str, Any] = dataclasses.asdict(config)
        del config_dict["eval_metric"]
        f.write(
            json.dumps(
                {
                    "config": config_dict,
                    "result": result,
                },
                ensure_ascii=False,
                cls=NumpyJsonEncoder,
            )
        )


if __name__ == "__main__":
    flags.mark_flag_as_required("train_path")
    flags.mark_flag_as_required("eval_path")
    flags.mark_flag_as_required("keyword_path")
    flags.mark_flag_as_required("title_path")
    flags.mark_flag_as_required("model_path")

    tf.compat.v1.app.run()
