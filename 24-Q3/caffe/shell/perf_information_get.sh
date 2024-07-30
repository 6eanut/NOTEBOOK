#! /bin/sh

# $1: Complete execution command    $2: store folder for generated file
# eg:   ./performance_counter.sh "./hackbench -s 512 -l 200 -g 15 -f 25 -P" /home

if [ $# -ne 2 ]; then
    echo "Usage:  ./performance_counter.sh parameter1 parameter2"
    exit 1
fi

echo "parameter1=$1"

result=$(echo "$1" | sed 's:.*/::')

file_name=$(echo "$result" | sed 's/ //g')
echo "file name : $file_name"

if [ -f "performance.txt" ]; then
    rm -f performance.txt
    echo "performance.txt has been deleted"
fi

perf stat --sync -e duration_time,task-clock,cycles,instructions,cache-references,cache-misses,branches,branch-misses,L1-dcache-loads,L1-dcache-load-misses,LLC-load-misses,LLC-loads -r 1 -o performance.txt $1

awk '{print $1, $2, $3}' performance.txt > performance_tmp.txt

mv performance_tmp.txt performance.txt

duration_time=`cat performance.txt | grep "duration_time" | awk '{print $1}' | sed 's/,//g'`

task_clock=`cat performance.txt | grep "task-clock" | awk '{print $1}' | sed 's/,//g'`

cpu_cycle=`cat performance.txt | grep "cycles" | awk '{print $1}' | sed 's/,//g'`

instruction=`cat performance.txt | grep "instructions" | awk '{print $1}' | sed 's/,//g'`

cache_references=`cat performance.txt | grep "cache-references" | awk '{print $1}' | sed 's/,//g'`

cache_misses=`cat performance.txt | grep "cache-misses" | awk '{print $1}' | sed 's/,//g'`

branches=`cat performance.txt | grep "branches" | awk '{print $1}' | sed 's/,//g'`

branch_misses=`cat performance.txt | grep "branch-misses" | awk '{print $1}' | sed 's/,//g'`

L1_dcache_loads=`cat performance.txt | grep "L1-dcache-loads" | awk '{print $1}' | sed 's/,//g'`

L1_dcache_load_misses=`cat performance.txt | grep "L1-dcache-load-misses" | awk '{print $1}' | sed 's/,//g'`

LLC_load_misses=`cat performance.txt | grep "LLC-load-misses" | awk '{print $1}' | sed 's/,//g'`

LLC_loads=`cat performance.txt | grep "LLC-loads" | awk '{print $1}' | sed 's/,//g'`

printf "\n\n"

echo "Avg 10 times duration time: $duration_time"

printf "Avg 10 times task clock: %.3f\n" $task_clock

echo "Avg 10 times cpu-cycles:   $cpu_cycle"

echo "Avg 10 times instructions: $instruction"

echo "Avg 10 times cache references: $cache_references"

echo "Avg 10 times cache misses: $cache_misses"

echo "Avg 10 times branches: $branches"

echo "Avg 10 times branch misses: $branch_misses"

echo "Avg 10 times L1 dcache loads: $L1_dcache_loads"

echo "Avg 10 times L1 dcache load misses: $L1_dcache_load_misses"

echo "Avg 10 times LLC load misses: $LLC_load_misses"

echo "Avg 10 times LLC load: $LLC_loads"

IPC=`echo "scale=3; $instruction / $cpu_cycle" | bc`
printf "Avg 10 times IPC: %.3f\n" $IPC

if [ -f "$file_name.txt" ]; then
    rm -f $file_name.txt
    echo "$file_name.txt has been deleted"
fi

echo $duration_time >> $file_name.txt
echo $task_clock >> $file_name.txt
echo $cpu_cycle >> $file_name.txt
echo $instruction >> $file_name.txt
echo $cache_references >> $file_name.txt
echo $cache_misses >> $file_name.txt
echo $branches >> $file_name.txt
echo $branch_misses >> $file_name.txt
echo $L1_dcache_loads >> $file_name.txt
echo $L1_dcache_load_misses >> $file_name.txt
echo $LLC_load_misses >> $file_name.txt
echo $LLC_loads >> $file_name.txt
printf "%.3f\n" $IPC >> $file_name.txt

cat $file_name.txt
mv $file_name.txt $2

rm -f performance.txt
