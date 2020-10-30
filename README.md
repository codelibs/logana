Logana: Log Analysis System
===========================

## Overview

Logana is Search/Recommend Log Analysis system.

### Kernel settings for elasticsearch

Logana server is elasticsearch cluster.
elasticsearch needs to set vm.max\_map\_count to at least 262144.
See [Install Elasticsearch with Docker](https://www.elastic.co/guide/en/elasticsearch/reference/current/docker.html#docker-prod-prerequisites).

## Usage

### Start Logana Server

```
$ ./bin/logana server start
```

If you want to start Logana with Kibana, add `-k` option:

```
$ ./bin/logana server start -k
```

### Stop Logana Server

```
$ ./bin/logana server stop
```

### Clean Up on Logana Server

```
$ ./bin/logana server clean
```

### Show Stats

```
$ ./bin/logana show
```
## Others

### Log Format

Logana defines a log format with dynamic templates for elasticsearch.
It's as below:

```
{
  "request": {
    "id": { // ID values
      "name": "value",
      ...
    },
    "attributes": { // attributes for request
      "keyword": { // field type
        "name": "value",
        ...
      },
      ... // available field types: keyworkd, text, integer, long, float, double, date, ip
    },
    "conditions": { // search conditions
      "keyword": { // field type
        "name": "value",
        ...
      },
      ... // available field types: keyworkd, text, integer, long, float, double, date, ip
    }
  },
  "response": {
    "attributes": { // attributes for response
      "keyword": { // field type
        "name": "value",
        ...
      },
      ... // available field types: keyworkd, text, integer, long, float, double, date, ip
    },
    "results": {
      "doc_X": { // viewed document
        "id": "value", // ID for document
        "keyword": { // field type
          "name": "value",
          ...
        },
        ... // available field types: keyworkd, text, integer, long, float, double, date, ip
      },
      ...
    }
  }
}
```

An example request is like this:

```
curl -H "Content-Type: application/json" -XPOST localhost:9200/logana_log.2020-10/_doc/12345 -d '
{
  "request": {
    "id": {
      "request": "12345",
      "session": "abcde"
    },
    "attributes": {
      "keyword": {
        "user_agent": "Mozilla/5.0 ...",
        "referer": "https:/..."
       },
      "date": {
        "request_time": "2020-09-06T00:00:06.166+0000"
       }
    },
    "conditions": {
      "keyword": {
        "query": "java",
        "location": "tokyo",
        "category": ["tech", "news"]
      }
    }
  },
  "response": {
    "attributes": {
      "integer": {
        "took": 10
      },
      "long": {
        "total_hits": 321
      }
    },
    "results": {
      "doc_1": {
        "id": "docid345",
        "keyword": {
          "title": "Java Programing",
          "category": "java"
         },
        "boolean": {
          "is_clicked": true
         },
        "date": {
          "click_time": "2020-09-06T00:01:16.234+0000"
         }
      },
      "doc_2": {
        "id": "docid534",
        "keyword": {
          "title": "Java Book",
          "category": "java"
         },
        "boolean": {
          "is_clicked": false
         },
        "date": {
         }
      },
      "doc_3": {
        "id": "docid675",
        "keyword": {
          "title": "Java News",
          "category": "news"
         },
        "boolean": {
          "is_clicked": true
         },
        "date": {
          "click_time": "2020-09-06T00:02:34.412+0000"
         }
      }
    }
  }
}'


