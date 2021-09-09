import sys
sys.path.append(".")
import json
import subprocess
#create_env_cmd = ["conda", "env", "create", "-f", "./env.yml", "--force"]
#subprocess.Popen(create_env_cmd).communicate()
activate_env = ["conda", "activate", "bug_ml"]
subprocess.Popen(activate_env).communicate()
from data_aggregation import get_all_changed_methods
from data_aggregation import match_reports_fixes
from data_aggregation import add_path_info
from data_aggregation import collect_sources
from data_aggregation.EDA import count_optimal_frames_limit


args = json.load(open('collect_data_properties.json', 'r'))
collect_commits = ["sudo", "sh", "./collect_fix_commits.sh", args['intellij_path']]
subprocess.Popen(collect_commits).communicate()

get_all_changed_methods.main(args['intellij_path'])
match_reports_fixes.main(args['reports_path'])
add_path_info.main(args['intellij_path'], args['reports_path'], args['files_limit'])
collect_sources.main(args['intellij_path'], args['reports_path'], args['files_limit'])
count_optimal_frames_limit.main(args['reports_path'])

prepare_code2seq_cmd = ["sh", "code2seq.sh"]
subprocess.Popen(prepare_code2seq_cmd).communicate()
