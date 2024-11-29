#!/bin/bash

run_command() {
  echo "Running command: $@"
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Command '$1' failed with exit status $status."
    exit $status
  fi
  return $status
}
