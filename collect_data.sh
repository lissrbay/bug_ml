
#!/bin/bash
embeddings_type=$1
path_to_intellij=$2
path_to_reports=$3
#FILE=./intellij-community
#if [ ! -d "$FILE" ]; then
#    git clone https://github.com/JetBrains/intellij-community
#fi

#echo "Intellij repository collected"
#conda env create -f ./env.yml --force
eval "$(conda shell.bash hook)"
conda activate bug_ml
touch conda_info.txt
truncate -s 0 conda_info.txt
conda info >> conda_info.txt
path_to_conda=$(grep -o 'active env location : .*$' conda_info.txt)
path_to_conda=($path_to_conda)
path_to_conda=${path_to_conda[4]}
path_to_python=${path_to_conda}'/bin/python'
echo 'Path to conda: '$path_to_conda
eval $path_to_python -m pip install rouge
eval $path_to_python -m pip install gensim
eval $path_to_python -m pip install catboost
cd ./data_aggregation
sudo sh ./collect_fix_commits.sh ${path_to_intellij}
$path_to_python get_all_changed_methods.py ${path_to_intellij}
$path_to_python match_reports_fixes.py ${path_to_reports}

$path_to_python add_path_info.py ${path_to_intellij} ${path_to_reports} 80
$path_to_python collect_sources.py ${path_to_intellij} ${path_to_reports} 80
$path_to_python ./EDA/count_optimal_frames_limit.py ${path_to_reports}
cd ..
echo $embeddings_type
if [[ "$embeddings_type" == "code2seq" ]]; then
    path_to_code2seq=./code2seq
    if [ ! -d "$path_to_code2seq" ]; then
        git clone https://github.com/tech-srl/code2seq.git
    fi
    rsync ./embeddings/code2seq ./code2seq
    cd $path_to_code2seq
    wget https://s3.amazonaws.com/code2seq/model/java-large/java-large-model.tar.gz
    tar -xvzf java-large-model.tar.gz
    $path_to_python code2seq.py --load models/java-large-model/model_iter52.release --predict --reports ${path_to_reports}
    cd ..
fi


if ["$embeddings_type" == "code2vec"]; then
    path_to_code2vec=./code2vec
    if [ ! -d "$path_to_code2vec" ]; then
        git clone https://github.com/tech-srl/code2vec.git
    fi
    rsync ./embeddings/code2vec ./code2vec
    cd $path_to_code2seq
    wget https://s3.amazonaws.com/code2vec/model/java14m_model.tar.gz
    tar -xvzf java14m_model.tar.gz
    $path_to_python code2vec.py --load models/java-large-model/model_iter52.release --predict
    cd ..
fi
mkdir ./data
$path_to_python embeddings/match_embeddings_with_methods.py ${path_to_reports} $embeddings_type 80
