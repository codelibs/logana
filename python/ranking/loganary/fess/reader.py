import collections
import glob
import gzip
import json
import logging
import os
from typing import Any, Dict, Generator, List, Optional

from loganary import ranking

logger = logging.getLogger(__name__)


class SearchLogReader(ranking.Reader):
    def __init__(self, log_dir: str, prefix: str = "searchlog") -> None:
        super().__init__()
        self._log_files: List[str] = sorted(
            glob.glob(f"{log_dir}/{prefix}*.log.gz"), key=os.path.getmtime
        )
        log_file: str = f"{log_dir}/{prefix}.log"
        if os.path.exists(log_file):
            self._log_files.append(log_file)

    @staticmethod
    def _open(filename: str, mode="rt", encoding="utf-8"):
        logger.info(f"Loading {filename}")
        if filename.endswith(".gz"):
            return gzip.open(filename, mode=mode, encoding=encoding)
        return open(filename, mode=mode, encoding=encoding)

    @staticmethod
    def _create_impression(data: Dict[str, Any]) -> Dict[str, Any]:
        results: Dict[str, Any] = {}
        if "documents" in data:
            for i, doc in enumerate(data["documents"]):
                results[f"doc_{i+1}"] = {
                    "id": doc.get("doc_id"),
                    "keyword": {
                        "filetype": doc.get("filetype"),
                        "url": doc.get("url"),
                        "site": doc.get("site"),
                        "filename": doc.get("filename"),
                        "host": doc.get("host"),
                        "digest": doc.get("digest"),
                        "mimetype": doc.get("mimetype"),
                        "lang": doc.get("lang"),
                    },
                    "boolean": {
                        "clicked": False,
                    },
                    "integer": {
                        "click_count": doc.get("click_count"),
                        "favorite_count": doc.get("favorite_count"),
                    },
                    "long": {
                        "content_length": doc.get("content_length"),
                    },
                    "float": {
                        "score": doc.get("score"),
                        "boost": doc.get("boost"),
                    },
                    "date": {
                        "created_time": doc.get("created"),
                        "last_modified": doc.get("last_modified"),
                        "timestamp": doc.get("timestamp"),
                    },
                }

        return {
            "request": {
                "id": {
                    "query": data.get("query_id"),
                },
                "attributes": {
                    "keyword": {
                        "access_type": data.get("access_type"),
                        "client_ip": data.get("client_ip"),
                        "user_agent": data.get("user_agent"),
                        "virtual_host": data.get("virtual_host"),
                    },
                },
                "conditions": {
                    "keyword": {
                        "search_word": data.get("search_word"),
                        "languages": data.get("languages"),
                        "roles": data.get("roles"),
                    },
                },
            },
            "response": {
                "results": results,
                "attributes": {
                    "keyword": {
                        "hit_count_relation": data.get("hit_count_relation"),
                    },
                    "integer": {
                        "query_offset": data.get("query_offset"),
                        "query_page_size": data.get("query_page_size"),
                    },
                    "long": {
                        "hit_count": data.get("hit_count"),
                        "query_time": data.get("query_time"),
                        "response_time": data.get("response_time"),
                    },
                },
            },
            "@timestamp": data.get("requested_at"),
        }

    @staticmethod
    def _update_impression(impression: Dict[str, Any], data: Dict[str, Any]) -> None:
        results: Dict[str, Dict[str, Any]] = impression["response"]["results"]
        doc_id: str = data["doc_id"]
        for _, doc in results.items():
            if doc_id == doc.get("id"):
                doc["boolean"]["clicked"] = True
                doc["date"]["clicked_time"] = doc.get("requested_at")
                return
        logger.warning(
            f'query_id:{data.get("query_id")} does not contain doc_id:{doc_id}.'
        )

    def readobjects(
        self, process_size: int = 100, queue_size: int = 200
    ) -> Generator[Dict[str, Dict[str, Any]], None, None]:
        impressions: collections.OrderedDict = collections.OrderedDict()
        for log_file in self._log_files:
            with self._open(log_file) as f:
                for line in f.readlines():
                    log_obj: Dict[str, Any] = json.loads(line)
                    event_type: Optional[str] = log_obj.get("event_type")
                    if event_type == "log":
                        ipression: Dict[str, Dict[str, Any]] = self._create_impression(
                            log_obj
                        )
                        impressions[ipression["request"]["id"]["query"]] = ipression
                        if len(impressions) > queue_size:
                            while len(impressions) > process_size:
                                _, impression = impressions.popitem()
                                yield impression
                    elif event_type == "click":
                        query_id: Optional[str] = log_obj.get("query_id")
                        if query_id in impressions:
                            impression = impressions.get(query_id)
                            self._update_impression(impression, log_obj)
                        else:
                            logger.warning(f"query_id:{query_id} is not found.")

        for _, impression in impressions.items():
            yield impression
