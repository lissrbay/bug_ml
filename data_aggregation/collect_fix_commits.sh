path_to_intellij = $1
touch commit_fix_hashes.txt
truncate -s 0 commit_fix_hashes.txt
curr_path = $(pwd)
cd ${path_to_intellij}
git log --grep="(^|\s)EA-[\d]+" -P >> ${curr_path}/data_aggregation/commit_fix_hashes.txt