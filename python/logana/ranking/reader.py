import hashlib
import logging
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, Generator, List, Set, Tuple

import numpy as np
import tensorflow as tf
from logana.common import deep_get
from logana.ranking.data import (TfRecordConfig, TfRecordField, add_example,
                                 create_elwc)

logger = logging.getLogger(__name__)


class Reader(ABC):

    @abstractmethod
    def readobjects(self, process_size: int = 100, queue_size: int = 200) -> Generator[Dict[str, Dict[str, Any]], None, None]:
        pass

    @staticmethod
    def _get_data_key(fields: List[TfRecordField], data: Dict[str, Any]):
        key: str = ''
        for field in fields:
            if field.primary:
                key += '\t'+deep_get(data, field.proerty_name)
        key = hashlib.md5(key.encode('utf-8')).hexdigest()
        return key

    def _get_relevance_stats(self, config: TfRecordConfig) -> Dict[str, Dict[str, Any]]:
        example_stats: Dict[str, Dict[str, Any]] = {}
        for impression in self.readobjects():
            conditions: Dict[str, Any] = deep_get(
                impression, 'request.conditions')
            example_key: str = self._get_data_key(
                config.example_fields, conditions)
            if example_key not in example_stats:
                example_stats[example_key] = {}
            context_stats: Dict[str, int] = example_stats.get(example_key)

            for result in deep_get(impression, 'response.results', {}).values():
                context_key: str = self._get_data_key(
                    config.context_fields, result)
                if context_key not in context_stats:
                    context_stats[context_key] = {'total': 0, 'action': 0}
                stats: Dict[str, int] = context_stats.get(context_key)
                stats['total'] += 1
                if deep_get(result, config.action_name, False):
                    stats['action'] += 1

        conversion_list: List[float] = []
        for context_stats in example_stats.values():
            for stats in context_stats.values():
                conversion_list.append(
                    float(stats['action']/float(stats['total'])))
        conversions: np.ndarray = np.array(conversion_list)
        thresholds: float = np.quantile(
            conversions, [x/config.relevance_level for x in range(1, config.relevance_level)])
        for context_stats in example_stats.values():
            for stats in context_stats.values():
                conversion_rate: float = float(
                    stats['action']/float(stats['total']))
                for i, t in enumerate(thresholds):
                    if conversion_rate <= t:
                        stats['relevance'] = i
                        break
                if 'relevance' not in stats:
                    stats['relevance'] = config.relevance_level - 1
        return example_stats

    def _get_example_list(self, config: TfRecordConfig) -> Dict[str, Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]]:
        relevance_stats: Dict[str, Dict[str, Any]
                              ] = self._get_relevance_stats(config)
        examples: Dict[str, Tuple[Dict[str, Any],
                                  Dict[str, Dict[str, Any]]]] = {}
        for impression in self.readobjects():
            condition: Dict[str, Any] = deep_get(
                impression, 'request.conditions')
            example_key: str = self._get_data_key(
                config.example_fields, condition)
            if example_key not in examples:
                example: Dict[str, Any] = {}
                for field in config.example_fields:
                    example[field.name] = field.analyzer(
                        deep_get(condition, field.proerty_name))
                examples[example_key] = (example, {})
            _, contexts = examples.get(example_key)
            for result in deep_get(impression, 'response.results', {}).values():
                context_key: str = self._get_data_key(
                    config.context_fields, result)
                if context_key in contexts:
                    continue
                total: int = relevance_stats[example_key][context_key]['total']
                action: int = relevance_stats[example_key][context_key]['action']
                if total <= config.min_total or action <= config.min_action:
                    continue
                context: Dict[str, Any] = {}
                for field in config.context_fields:
                    context[field.name] = field.analyzer(
                        deep_get(result, field.proerty_name))
                context['relevance'] = relevance_stats[example_key][context_key]['relevance']
                contexts[context_key] = context
        return examples

    def to_tfrecords(self, config: TfRecordConfig) -> None:
        train_relevance_count: np.ndarray = np.zeros(
            config.relevance_level, dtype=np.int)
        eval_relevance_count: np.ndarray = np.zeros(
            config.relevance_level, dtype=np.int)
        count: int = 0
        vocabs: Dict[str, Set[str]] = {}

        for field in config.example_fields:
            if field.dictionary is not None:
                vocabs[field.name] = set()
        for field in config.context_fields:
            if field.dictionary is not None:
                vocabs[field.name] = set()

        def _add_example_to_context(elwc: Any, context: Dict[str, Any]) -> None:
            context_fields: Dict[str, tf.train.Feature] = {}
            for field in config.context_fields:
                context_fields[field.name] = field.featurizer(
                    context[field.name])
                if field.dictionary is not None:
                    for token in context[field.name]:
                        vocabs[field.name].add(token)
            add_example(elwc, context['relevance'], context_fields)

        with tf.io.TFRecordWriter(config.train_path) as train_writer, tf.io.TFRecordWriter(config.eval_path) as test_writer:
            for (example, contexts) in self._get_example_list(config).values():
                if len(contexts) == 0:
                    continue

                example_fields: Dict[str, tf.train.Feature] = {}
                for field in config.example_fields:
                    example_fields[field.name] = field.featurizer(
                        example[field.name])
                    if field.dictionary is not None:
                        for token in example[field.name]:
                            vocabs[field.name].add(token)

                train_elwc: Any = create_elwc(example_fields)
                test_elwc: Any = create_elwc(example_fields)

                contexts: List[Dict[str, Any]] = [x for x in contexts.values()]
                random.shuffle(contexts)

                train_contexts: List[Dict[str, Any]] = contexts[int(
                    len(contexts)*config.split_ratio):]
                for context in train_contexts:
                    _add_example_to_context(train_elwc, context)
                    train_relevance_count[context['relevance']] += 1

                eval_contexts: List[Dict[str, Any]] = contexts[0:int(
                    len(contexts)*config.split_ratio)]
                for context in eval_contexts:
                    _add_example_to_context(test_elwc, context)
                    eval_relevance_count[context['relevance']] += 1

                logger.debug(
                    f'Train: {"/".join([str(x) for x in train_relevance_count])}={sum(train_relevance_count)},' +
                    f' Eval: {"/".join([str(x) for x in eval_relevance_count])}={sum(eval_relevance_count)}')

                count += 1
                if count % 500 == 0:
                    logger.info(
                        f'query : {count}, train: {sum(train_relevance_count)}, eval: {sum(eval_relevance_count)}')

                train_writer.write(train_elwc.SerializeToString())
                if len(eval_contexts) > 0:
                    test_writer.write(test_elwc.SerializeToString())

        for field in config.example_fields:
            if field.dictionary is not None:
                with open(field.dictionary, 'wt', encoding='utf-8') as f:
                    for x in sorted(vocabs[field.name]):
                        f.write(x)
                        f.write('\n')

        for field in config.context_fields:
            if field.dictionary is not None:
                with open(field.dictionary, 'wt', encoding='utf-8') as f:
                    for x in sorted(vocabs[field.name]):
                        f.write(x)
                        f.write('\n')
        logger.info(
            f'query : {count}, train: {sum(train_relevance_count)}, eval: {sum(eval_relevance_count)}')