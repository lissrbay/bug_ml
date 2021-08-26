path_to_code2seq=./code2seq
if [ ! -d "$path_to_code2seq" ]; then
    git clone https://github.com/tech-srl/code2seq.git
fi
rsync ./embeddings/code2seq ./code2seq
cd $path_to_code2seq
wget https://s3.amazonaws.com/code2seq/model/java-large/java-large-model.tar.gz
tar -xvzf java-large-model.tar.gz
cd ..