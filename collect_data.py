import json
import subprocess
import data_aggregation.get_all_changed_methods
import data_aggregation.match_reports_fixes
import data_aggregation.add_path_info
import data_aggregation.EDA.count_optimal_frames_limit

create_env_cmd = ["conda", "env", "create", "-f", "./env.yml", "--force"]
subprocess.Popen(create_env_cmd).communicate()

activate_env = ["conda", "activate", "bug_ml"]
subprocess.Popen(create_env_cmd).communicate()

args = json.load(open('collect_data_properties.json', 'r'))

collect_commits = ["sudo", "sh", "./collect_fix_commits.sh", args.path_to_intellij]
subprocess.Popen(collect_commits).communicate()

get_all_changed_methods.main()
match_reports_fixes.main(args.reports_path)
add_path_info.main(args.intellij_path, args.reports_path, args.files_limit)
collect_sources.main(args.intellij_path, args.reports_path, args.files_limit)
count_optimal_frames_limit.main(args.reports_path)

prepare_code2seq_cmd = ["sh", "code2seq.sh"]
subprocess.Popen(prepare_code2seq_cmd).communicate()

code2seq_cmd = ["python", "code2seq.py", "--load", "models/java-large-model/model_iter52.release", "--predict", "--reports", args.reports_path]
subprocess.Popen(code2seq_cmd).communicate()

baseline_cmd = ["python", "models/baseline.py", args.reports_path]
subprocess.Popen(baseline_cmd).communicate()