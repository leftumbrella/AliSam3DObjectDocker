#!/bin/sh

set -eu

INITIALIZER_TIMEOUT="${FC_INITIALIZER_HTTP_TIMEOUT:-295}"
PUBLIC_PORT="${PORT:-9000}"
SAM3_PORT="${SAM3_INTERNAL_PORT:-9001}"
STARTED_AT="$(date +%s)"

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

validate_port() {
  name="$1"
  value="$2"
  case "$value" in
    ''|*[!0-9]*)
      printf '%s must be an integer between 1 and 65535\n' "$name" >&2
      exit 2
      ;;
  esac
  if [ "$value" -le 0 ] || [ "$value" -gt 65535 ]; then
    printf '%s must be an integer between 1 and 65535\n' "$name" >&2
    exit 2
  fi
}

remaining_timeout() {
  now="$(date +%s)"
  remaining=$((INITIALIZER_TIMEOUT - now + STARTED_AT))
  if [ "$remaining" -le 0 ]; then
    printf 'FC Initializer exceeded its %s second budget\n' "$INITIALIZER_TIMEOUT" >&2
    exit 1
  fi
  printf '%s\n' "$remaining"
}

warm_model() {
  model="$1"
  url="$2"

  while :; do
    remaining="$(remaining_timeout)"
    printf '[initializer] warming %s\n' "$model"
    if curl \
      --fail \
      --silent \
      --show-error \
      --connect-timeout 1 \
      --max-time "$remaining" \
      --request POST \
      "$url"; then
      printf '\n[initializer] %s ready\n' "$model"
      return 0
    else
      curl_status="$?"
    fi

    if [ "$curl_status" -ne 7 ]; then
      return "$curl_status"
    fi
    sleep 1
  done
}

validate_port PORT "$PUBLIC_PORT"
validate_port SAM3_INTERNAL_PORT "$SAM3_PORT"

warm_model SAM3 "http://127.0.0.1:${SAM3_PORT}/_fc/warmup"
warm_model SAM3D "http://127.0.0.1:${PUBLIC_PORT}/_fc/warmup"
