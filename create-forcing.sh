#!/bin/bash
seed=$(date +'%d%m%Y')
amp=(5 10 15 20 25)
for A in ${amp[*]}; do
  filename="forcing-${A}mm-${seed}.csv"
  echo "creating ${filename}..."
  python3 ./CreateRandomForcing_v2.py --seed $seed $A 4 30 8000 ${filename}
done
