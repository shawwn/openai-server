#!/bin/bash
set -ex
exec python3 -m openai_server "$@"

