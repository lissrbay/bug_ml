path_to_repo=$1
data_dir=$2
if [ -e "$data_dir" ]; then echo "data dir already exist"; else mkdir "$data_dir"; fi;
if [ -e "$data_dir/commit_fix_hashes.txt" ]; then echo "commit info already exist"; else
  touch "$data_dir/commit_fix_hashes.txt";
  truncate -s 0 "$data_dir/commit_fix_hashes.txt";
  git -C "$path_to_repo" log --grep="(^|\s)EA-[\d]+" -P -- "$path_to_repo" >> "$data_dir/commit_fix_hashes.txt";
fi;