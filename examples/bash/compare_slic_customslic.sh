#!/bin/bash
# Copyright (c) 2016, David Stutz
# Contact: david.stutz@rwth-aachen.de, davidstutz.de
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its contributors
#    may be used to endorse or promote products derived from this software
#    without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
# THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Example of evaluating and comparing SEEDS and reSEEDS
# Supposed to be run from within examples/.

algo1="slic"
algo2="customslic"

echo "Comparig ${algo1} and ${algo2}"

SUPERPIXELS=("200" "300" "400" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2400" "2800" "3200" "3600" "4000" "4600" "5200")

rm -rf ../output/${algo1}
rm -rf ../output/${algo2}
rm ../output/${algo1}*
rm ../output/${algo2}*


for SUPERPIXEL in "${SUPERPIXELS[@]}"
do
    # algo 1
    ../bin/${algo1}_cli ../data/BSDS500/images/test/ --superpixels $SUPERPIXEL -o ../output/${algo1}/$SUPERPIXEL -w --iterations 3
    ../bin/eval_summary_cli ../output/${algo1}/$SUPERPIXEL ../data/BSDS500/images/test ../data/BSDS500/csv_groundTruth/test --append-file ../output/${algo1}.csv --vis
    find ../output/${algo1}/$SUPERPIXEL -type f -name '*[^summary|correlation|results].csv' -delete
    
    # algo 2
    ../bin/${algo2}_cli ../data/BSDS500/images/test/ --superpixels $SUPERPIXEL -o ../output/${algo2}/$SUPERPIXEL -w --iterations 3
    ../bin/eval_summary_cli ../output/${algo2}/$SUPERPIXEL ../data/BSDS500/images/test ../data/BSDS500/csv_groundTruth/test --append-file ../output/${algo2}.csv --vis
    find ../output/${algo2}/$SUPERPIXEL -type f -name '*[^summary|correlation|results].csv' -delete
done

../bin/eval_average_cli ../output/${algo1}.csv -o ../output/${algo1}_average.csv
../bin/eval_average_cli ../output/${algo2}.csv -o ../output/${algo2}_average.csv

# run python script to create graphs of the comparison
python ./make-graph.py  ${algo1} ${algo2}