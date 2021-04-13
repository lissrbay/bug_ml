touch commit_fix_hashes.txt
truncate -s 0 commit_fix_hashes.txt
cd ../intellij
git log --grep="(^|\s)EA-[\d]+" -P >> ../data_aggregation/commit_fix_hashes.txt