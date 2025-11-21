# $1 cuda $2 filename $3 suffix
nohup python $2.py -c $1 > $2$3.log 2>&1 &
