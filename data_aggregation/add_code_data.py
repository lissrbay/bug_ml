import os
from tqdm import tqdm
import json
import re
from parser_java_kotlin import Parser
from add_path_info import load_report
import pickle

def remove_tabs(code):
    code = list(filter(lambda x: not (x.strip()[:2] == '//'), code))
    code = '\n'.join(code)
    code = re.sub(' +', ' ', code)
    return re.sub('\t+', '', code)


def code_fragment( bounds, code):
    if not bounds:
        return ''
    if bounds[1]<= bounds[0]:
        return ''
    return ''.join(code)[bounds[0]: bounds[1]]


def collect_code(path):
    reports_code = []
    reports_ids = []
    targets = []
    for root, dirs, files in tqdm(os.walk(path)):
        if (root == path):
            continue
        report = load_report(root + '.json')
        all_methods_code = []
        reports_ids.append(int(root.split('/')[-1]))
        target = []
        for frame in report['frames'][:80]:
            target.append(frame['label'])
            try:
                filename = frame['file_name']
                method_name = frame['method_name'].split('.')[-1]
                parser = Parser()
                txt = open(root+'/'+filename)
                txt = remove_tabs(txt)
                ast = parser.parse(txt, filename)
                method_info = ast.get_method_names_and_bounds()
                code = ''
                for name, bounds in method_info:
                    name_ = name.split(':')[-1]
                    if method_name in name:
                        method_code = code_fragment(bounds[0], txt)
                        code = name_ + method_code
                all_methods_code.append(code)
            except Exception:
                all_methods_code.append(code)
        targets.append(target)
        reports_code.append(all_methods_code)
    return reports_code, targets, reports_ids

PATH_TO_REPORTS = os.path.join("..", "intellij_fixed_201007")

if __name__ == "__main__":
    path = os.path.join(PATH_TO_REPORTS, "labeled_reports")
    path_save = os.path.join("..", "data")
    reports_code, targets, reports_ids = collect_code(path)
    pickle.dump(reports_code, open(os.path.join(path_save, "reports_code")))
    pickle.dump(targets, open(os.path.join(path_save, "targets")))
    pickle.dump(reports_ids, open(os.path.join(path_save, "reports_ids")))

