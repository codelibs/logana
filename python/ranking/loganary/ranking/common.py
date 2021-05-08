import json
import logging
import os
import random
import unicodedata
from functools import reduce
from typing import Callable, Dict, List

import fugashi
import numpy as np
import tensorflow_ranking as tfr

logger = logging.getLogger(__name__)


def japanese_tokenizer(
    tagger: fugashi.Tagger,
    text: str,
    pos_filter=lambda x: x in ["名詞"],
    empty_str: str = "__EMPTY__",
) -> List[str]:
    text = unicodedata.normalize("NFKC", text).lower().strip()
    if len(text) > 0:
        tokens: List[str] = []
        for word in tagger(text):
            pos = word.pos.split(",")[0]
            if pos_filter(pos):
                tokens.append(word.surface)
        return tokens
    else:
        return [empty_str]


class NumpyJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif callable(obj):
            return str(obj)
        else:
            try:
                return super(NumpyJsonEncoder, self).default(obj)
            except Exception as e:
                logger.warn(f"{obj} is not serializable: {str(e)}")
                return str(obj)


def deep_get(dictionary, keys, default=None):
    return reduce(
        lambda d, key: d.get(key, default) if isinstance(d, dict) else default,
        keys.split("."),
        dictionary,
    )


def setup_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
        tf.compat.v1.set_random_seed(seed)
    except ModuleNotFoundError:
        pass  # ignore
    os.environ["PYTHONHASHSEED"] = str(seed)


def setup_logging(verbose: bool) -> None:
    logging.getLogger().setLevel(level=logging.DEBUG if verbose else logging.INFO)
    try:
        import tensorflow as tf

        tf.compat.v1.logging.set_verbosity(
            tf.compat.v1.logging.DEBUG if verbose else tf.compat.v1.logging.INFO
        )
        tf.get_logger().propagate = False
    except ModuleNotFoundError:
        pass  # ignore


def get_ndcg_metric(topn: List[int] = [10, 20, 30, 40, 50]) -> Dict[str, Callable]:
    return {
        "metric/ndcg@%d"
        % x: tfr.metrics.make_ranking_metric_fn(
            tfr.metrics.RankingMetricKey.NDCG, topn=x
        )
        for x in topn
    }
