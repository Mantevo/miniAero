#!/bin/bash
echo "Running serial Ramp test."
../../make/src/miniaero.exe &> /dev/null
../tools/numeric_text_diff --relative-tolerance=1e-2 --floor=1e-3 results.0 results.gold > diff.txt
ESTATUS=$?
rm results.0 gradients.0 limiters.0 setupmesh.0 
exit $ESTATUS
