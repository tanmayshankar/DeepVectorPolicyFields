#!/bin/bash

set -e
set -o pipefail

for i in 0 1 2 3 4 5; do
  pushd "models/$i" > /dev/null
  echo "Starting $i..."
  python2 ../../BPRCNN_DesiredSimAll.py "$i" > stdout &
  popd > /dev/null
done

wait
