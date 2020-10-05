touch commit_hashes.txt
truncate -s 0 commit_hashes.txt
cd ./intellij-community
git log --grep="^EA-" --pretty=format:%H >> ../commit_hashes.txt