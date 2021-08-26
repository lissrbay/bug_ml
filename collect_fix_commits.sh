path_to_intellij=$1
if [ -e ./data ]; then echo "data dir already exist"; else mkdir ./data; fi;
cd ./data
curr_path=$(pwd)
touch $curr_path"/commit_fix_hashes.txt"
truncate -s 0 $curr_path"/commit_fix_hashes.txt"
cd ${path_to_intellij}
git log --grep="(^|\s)EA-[\d]+" -P >> ${curr_path}'/commit_fix_hashes.txt'