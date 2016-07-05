#!/bin/bash
echo "Running serial 3D sod test."
../../make/src/miniaero.exe &> /dev/null
../tools/numeric_text_diff results.0 results.gold > diff.txt
ESTATUS=$?
rm results.0 gradients.0 limiters.0 setupmesh.0 
exit $ESTATUS
