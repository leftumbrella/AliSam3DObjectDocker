#!/bin/sh

set -eu

INITIALIZER_URL="${FC_INITIALIZER_URL:-http://127.0.0.1:9000/initialize}"
INITIALIZER_TIMEOUT="${FC_INITIALIZER_HTTP_TIMEOUT:-295}"

case "$INITIALIZER_TIMEOUT" in
  ''|*[!0-9]*)
    printf 'FC_INITIALIZER_HTTP_TIMEOUT must be a positive integer\n' >&2
    exit 2
    ;;
esac

if [ "$INITIALIZER_TIMEOUT" -le 0 ] || [ "$INITIALIZER_TIMEOUT" -gt 295 ]; then
  printf 'FC_INITIALIZER_HTTP_TIMEOUT must be between 1 and 295 seconds\n' >&2
  exit 2
fi

exec curl \
  --fail \
  --silent \
  --show-error \
  --max-time "$INITIALIZER_TIMEOUT" \
  --request POST \
  "$INITIALIZER_URL"
