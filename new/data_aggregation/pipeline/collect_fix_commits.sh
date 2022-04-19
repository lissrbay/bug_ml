repo_path=$1
data_dir=$2
if [ -e "$data_dir" ]; then echo "Data dir already exist"; else mkdir "$data_dir"; fi;
if [ -e "$data_dir/fix_commits.txt" ]; then echo "Commit info already exist"; else
  touch "$data_dir/fix_commits.txt";
  truncate -s 0 "$data_dir/fix_commits.txt";
  git -C "$repo_path" log --grep="(^|\s)EA-[\d]+" -P -- "$repo_path" >> "$data_dir/fix_commits.txt";
fi;