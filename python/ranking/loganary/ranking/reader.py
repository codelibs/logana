import gzip
import hashlib
import json
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generator, List, Set, Tuple, ValuesView

import numpy as np
import tensorflow as tf
from loganary.ranking.common import deep_get
from loganary.ranking.data import (
    TfRecordConfig,
    TfRecordField,
    add_example,
    create_elwc,
)

logger = logging.getLogger(__name__)


class Reader(ABC):
    @abstractmethod
    def readobjects(
        self, process_size: int = 100, queue_size: int = 200
    ) -> Generator[Dict[str, Dict[str, Any]], None, None]:
        pass

    @staticmethod
    def _get_data_key(fields: List[TfRecordField], data: Dict[str, Any]):
        key: str = ""
        for field in fields:
            if field.primary:
                key += "\t" + str(deep_get(data, field.proerty_name))
        key = hashlib.md5(key.encode("utf-8")).hexdigest()
        return key

    def _get_relevance_stats(self, config: TfRecordConfig) -> Dict[str, Dict[str, Any]]:
        example_stats: Dict[str, Dict[str, Any]] = {}
        for impression in self.readobjects():
            conditions: Dict[str, Any] = deep_get(impression, "request.conditions")
            example_key: str = self._get_data_key(config.example_fields, conditions)
            if example_key not in example_stats:
                example_stats[example_key] = {}
            context_stats: Dict[str, Dict[str, int]] = example_stats[example_key]

            for result in deep_get(impression, "response.results", {}).values():
                context_key: str = self._get_data_key(config.context_fields, result)
                if context_key not in context_stats:
                    context_stats[context_key] = {"total": 0, "action": 0}
                stats: Dict[str, int] = context_stats[context_key]
                stats["total"] += 1
                if deep_get(result, config.action_name, False):
                    stats["action"] += 1

        conversion_list: List[float] = []
        for context_stats in example_stats.values():
            for stats in context_stats.values():
                conversion_list.append(float(stats["action"] / float(stats["total"])))
        conversions: np.ndarray = np.array(conversion_list)
        thresholds: List[float] = np.quantile(
            conversions,
            [x / config.relevance_level for x in range(1, config.relevance_level)],
        )
        for context_stats in example_stats.values():
            for stats in context_stats.values():
                conversion_rate: float = float(stats["action"] / float(stats["total"]))
                for i, t in enumerate(thresholds):
                    if conversion_rate <= t:
                        stats["relevance"] = i
                        break
                if "relevance" not in stats:
                    stats["relevance"] = config.relevance_level - 1
        return example_stats

    def _get_example_list(
        self, config: TfRecordConfig
    ) -> ValuesView[Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
        relevance_stats: Dict[str, Dict[str, Any]] = self._get_relevance_stats(config)
        examples: Dict[str, Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]] = {}
        for impression in self.readobjects():
            condition: Dict[str, Any] = deep_get(impression, "request.conditions")
            example_key: str = self._get_data_key(config.example_fields, condition)
            if example_key not in examples:
                example: Dict[str, Any] = {}
                original_example: Dict[str, Any] = {}
                for field in config.example_fields:
                    value: Any = deep_get(condition, field.proerty_name)
                    example[field.name] = field.analyzer(value)
                    if config.to_ndjson:
                        original_example[field.name] = value
                if config.to_ndjson:
                    example["_original"] = original_example
                examples[example_key] = (example, {})
            _, contexts = examples[example_key]
            for result in deep_get(impression, "response.results", {}).values():
                context_key: str = self._get_data_key(config.context_fields, result)
                if context_key in contexts:
                    continue
                total: int = relevance_stats[example_key][context_key]["total"]
                action: int = relevance_stats[example_key][context_key]["action"]
                if total <= config.min_total or action <= config.min_action:
                    continue
                context: Dict[str, Any] = {}
                original_context: Dict[str, Any] = {}
                for field in config.context_fields:
                    value = deep_get(result, field.proerty_name)
                    context[field.name] = field.analyzer(value)
                    if config.to_ndjson:
                        original_context[field.name] = value
                if "relevance" not in context:
                    relevance: int = relevance_stats[example_key][context_key][
                        "relevance"
                    ]
                    context["relevance"] = relevance
                    if config.to_ndjson:
                        original_context["relevance"] = relevance
                        context["_original"] = original_context
                contexts[context_key] = context
        return examples.values()

    @staticmethod
    def _write_examples_as_json(
        writer, example: Dict[str, Any], contexts: List[Dict[str, Any]]
    ) -> None:
        data: Dict[str, Any] = example.copy()
        data["contexts"] = contexts
        writer.write(json.dumps(data, ensure_ascii=False))
        writer.write("\n")

    @staticmethod
    def _add_example_to_context(
        config: TfRecordConfig,
        vocabs: Dict[str, Set[str]],
        elwc: Any,
        context: Dict[str, Any],
    ) -> None:
        context_fields: Dict[str, tf.train.Feature] = {}
        for field in config.context_fields:
            context_fields[field.name] = field.featurizer(context[field.name])
            if field.dictionary is not None:
                for token in context[field.name]:
                    vocabs[field.name].add(token)
        add_example(elwc, context["relevance"], context_fields)

    def to_tfrecords(self, config: TfRecordConfig) -> None:
        train_relevance_count: np.ndarray = np.zeros(
            config.relevance_level, dtype=np.int
        )
        eval_relevance_count: np.ndarray = np.zeros(
            config.relevance_level, dtype=np.int
        )
        count: int = 0
        vocabs: Dict[str, Set[str]] = {}

        for field in config.example_fields:
            if field.dictionary is not None:
                vocabs[field.name] = set()
        for field in config.context_fields:
            if field.dictionary is not None:
                vocabs[field.name] = set()

        with tf.io.TFRecordWriter(
            config.train_path
        ) as train_writer, tf.io.TFRecordWriter(
            config.eval_path
        ) as test_writer, gzip.open(
            f"{config.train_path}.ndjson.gz", mode="wt", encoding="utf-8"
        ) as train_json_writer, gzip.open(
            f"{config.eval_path}.ndjson.gz", mode="wt", encoding="utf-8"
        ) as eval_json_writer:
            for (example, contexts) in self._get_example_list(config):
                if len(contexts) == 0:
                    continue

                example_fields: Dict[str, tf.train.Feature] = {}
                for field in config.example_fields:
                    example_fields[field.name] = field.featurizer(example[field.name])
                    if field.dictionary is not None:
                        for token in example[field.name]:
                            vocabs[field.name].add(token)

                train_elwc: Any = create_elwc(example_fields)
                test_elwc: Any = create_elwc(example_fields)

                context_list: List[Dict[str, Any]] = [x for x in contexts.values()]
                random.shuffle(context_list)

                train_contexts: List[Dict[str, Any]] = context_list[
                    int(len(context_list) * config.split_ratio) :
                ]
                for context in train_contexts:
                    self._add_example_to_context(config, vocabs, train_elwc, context)
                    train_relevance_count[context["relevance"]] += 1
                if config.to_ndjson:
                    self._write_examples_as_json(
                        train_json_writer,
                        example["_original"],
                        [x["_original"] for x in train_contexts],
                    )

                eval_contexts: List[Dict[str, Any]] = context_list[
                    0 : int(len(context_list) * config.split_ratio)
                ]
                for context in eval_contexts:
                    self._add_example_to_context(config, vocabs, test_elwc, context)
                    eval_relevance_count[context["relevance"]] += 1
                if config.to_ndjson and len(eval_contexts) > 0:
                    self._write_examples_as_json(
                        eval_json_writer,
                        example["_original"],
                        [x["_original"] for x in eval_contexts],
                    )

                logger.debug(
                    f'Train: {"/".join([str(x) for x in train_relevance_count])}={sum(train_relevance_count)},'
                    + f' Eval: {"/".join([str(x) for x in eval_relevance_count])}={sum(eval_relevance_count)}'
                )

                count += 1
                if count % 500 == 0:
                    logger.info(
                        f"query : {count}, train: {sum(train_relevance_count)}, eval: {sum(eval_relevance_count)}"
                    )

                train_writer.write(train_elwc.SerializeToString())
                if len(eval_contexts) > 0:
                    test_writer.write(test_elwc.SerializeToString())

        for field in config.example_fields:
            if field.dictionary is not None:
                with open(field.dictionary, "wt", encoding="utf-8") as f:
                    for x in sorted(vocabs[field.name]):
                        f.write(x)
                        f.write("\n")

        for field in config.context_fields:
            if field.dictionary is not None:
                with open(field.dictionary, "wt", encoding="utf-8") as f:
                    for x in sorted(vocabs[field.name]):
                        f.write(x)
                        f.write("\n")
        logger.info(
            f"query : {count}, train: {sum(train_relevance_count)}, eval: {sum(eval_relevance_count)}"
        )

    def read_with_tfrecords(self, config: TfRecordConfig) -> Generator[Any, None, None]:
        vocabs: Dict[str, Set[str]] = {}

        for field in config.example_fields:
            if field.dictionary is not None:
                vocabs[field.name] = set()
        for field in config.context_fields:
            if field.dictionary is not None:
                vocabs[field.name] = set()

        for (example, contexts) in self._get_example_list(config):
            if len(contexts) == 0:
                continue

            example_fields: Dict[str, tf.train.Feature] = {}
            for field in config.example_fields:
                example_fields[field.name] = field.featurizer(example[field.name])
                if field.dictionary is not None:
                    for token in example[field.name]:
                        vocabs[field.name].add(token)

            elwc: Any = create_elwc(example_fields)

            context_list: List[Dict[str, Any]] = [x for x in contexts.values()]
            for context in context_list:
                self._add_example_to_context(config, vocabs, elwc, context)

            yield (example, context_list, elwc)


class NdjsonReader(Reader):
    def __init__(self, log_file: str, data_converter: Callable) -> None:
        super().__init__()
        self._log_file = log_file
        self._data_converter = data_converter

    @staticmethod
    def _open(filename: str, mode="rt", encoding="utf-8"):
        logger.info(f"Loading {filename}")
        if filename.endswith(".gz"):
            return gzip.open(filename, mode=mode, encoding=encoding)
        return open(filename, mode=mode, encoding=encoding)

    def readobjects(
        self, process_size: int = 100, queue_size: int = 200
    ) -> Generator[Dict[str, Dict[str, Any]], None, None]:
        count: int = 1
        with self._open(self._log_file) as f:
            for line in f.readlines():
                log_obj: Dict[str, Any] = json.loads(line)
                log_obj["_id"] = str(count)
                yield self._data_converter(log_obj)
                count += 1
