#!/bin/bash
echo "Running parallel ramp test using 16 processors."
mpirun -np 16 $1 &> /dev/null
diff=0
for i in `seq 0 15`;
do 
  ../tools/numeric_text_diff --relative-tolerance=1e-2 --floor=1e-3 results.$i results.$i.gold > diff.$i.txt
  diff=$(($diff + $?))
done
if [ $diff -gt 0 ];
then
  ESTATUS=1
else
  ESTATUS=0
fi

rm results.? gradients.? limiters.? setupmesh.?
rm results.?? gradients.?? limiters.?? setupmesh.??
exit $ESTATUS
