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

### Stop Logana Server

```
$ ./bin/logana server stop
```

