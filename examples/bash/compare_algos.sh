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

# Helper functions...

# list_include_item "10 11 12" "2"
function list_include_item {
  declare -a list=("${!1}")
  local item="$2"
  if [[ ${list[@]} =~ (^|[[:space:]])"$item"($|[[:space:]]) ]] ; then
    # yes, list include item
    result=0
  else
    result=1
  fi
  return $result
}

# Main program...
# Supposed to be run from within examples/.

#tuple = <algo>-<iterations>-<tile-size>-tuple

#TUPLES_TO_UPDATE_LIST=("customslic-3-0-8*8*1-tuple")
#TUPLES_TO_UPDATE_LIST=("slic-10")

SUPERPIXELS=("200" "300" "400" "600" "800" "1000" "1200" "1400" "1600" "1800" "2000" "2400" "2800" "3200" "3600" "4000" "4600" "5200")
NUM_ITERATIONS_LIST=("3")
SQUARE_SIDES_LIST=("0")
PYRAMID_PATTERN_LIST=("16*8*1" "8*8*1" "8*4*1" "8*2*1")

# Create algo_tuples list and print it.
algo_tuples_list=("slic-2" "slic-3" "slic-10")
for SQUARE_SIDES in "${SQUARE_SIDES_LIST[@]}"
do
	for NUM_ITERATIONS in "${NUM_ITERATIONS_LIST[@]}"
	do
		for PYRAMID_PATTERN in "${PYRAMID_PATTERN_LIST[@]}"
		do
			algo_tuple="customslic-$NUM_ITERATIONS-$SQUARE_SIDES-$PYRAMID_PATTERN-tuple"
			algo_tuples_list+=(${algo_tuple})
		done
	done
done
echo "***Algo tuples:***"
printf '%s\n' "${algo_tuples_list[@]}"
echo 

# Make a string of tuples as a Comma separated list
algo_tuples_string=""
for algo_tuple in "${algo_tuples_list[@]}"
do
	if [ -z "$algo_tuples_string" ]; then
    	algo_tuples_string="${algo_tuple}"
    else
    	algo_tuples_string="${algo_tuples_string},${algo_tuple}"
	fi
done

# Update all tuples if TUPLES_TO_UPDATE_LIST is not set.
if [ -z ${TUPLES_TO_UPDATE_LIST} ]; then 
	TUPLES_TO_UPDATE_LIST="${algo_tuples_list[@]}"
fi


# Main loop
for algo_tuple in "${algo_tuples_list[@]}"
do
	# Check if we want to run this tuple again
	if list_include_item TUPLES_TO_UPDATE_LIST[@] "${algo_tuple}" ; then
		echo
		echo
		echo "**Working on ${algo_tuple}"
	else 
	  	continue
	fi
	
	# Clean up old files
	rm -rf ../output/${algo_tuple}
	rm -f ../output/${algo_tuple}*

	IFS='-' read -r -a parts <<< "$algo_tuple"

	algo=${parts[0]}
	iterations=${parts[1]}
	tile_size=${parts[2]}
	pyramid_pattern=${parts[3]}

	for SUPERPIXEL in "${SUPERPIXELS[@]}"
	do
		echo "${algo_tuple} _ SP=$SUPERPIXEL"
		# Run the algo
		if [ "$algo" == "slic" ]; then

		    ../bin/${algo}_cli ../data/BSDS500/images/test/ --superpixels $SUPERPIXEL -o ../output/${algo_tuple}/$SUPERPIXEL -w --iterations $iterations

		elif [ "$algo" == "customslic" ]; then

			../bin/${algo}_cli ../data/BSDS500/images/test/ \
				--superpixels $SUPERPIXEL -o ../output/${algo_tuple}/$SUPERPIXEL -w \
				--iterations $iterations \
				--tile-size $tile_size \
				--pyramid-pattern $pyramid_pattern

		fi

		# Collect results
	    ../bin/eval_summary_cli ../output/${algo_tuple}/$SUPERPIXEL ../data/BSDS500/images/test ../data/BSDS500/csv_groundTruth/test --append-file ../output/${algo_tuple}.csv --vis
	    find ../output/${algo_tuple}/$SUPERPIXEL -type f -name '*[^summary|correlation|results].csv' -delete
	done

	# Average results
	../bin/eval_average_cli ../output/${algo_tuple}.csv -o ../output/${algo_tuple}_average.csv

done


# run python script to create graphs of the comparison
python ./make-graph.py ${algo_tuples_string}