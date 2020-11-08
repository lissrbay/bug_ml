import subprocess
from common import PathContextInformation


class Extractor:
    def __init__(self, config, jar_path, max_path_length, max_path_width):
        self.config = config
        self.max_path_length = max_path_length
        self.max_path_width = max_path_width
        self.jar_path = jar_path

    def extract_paths(self, path):
        command = ['java', '-cp', self.jar_path, 'JavaExtractor.App', '--max_path_length',
                   str(self.max_path_length-1), '--max_path_width', str(self.max_path_width), '--file', path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = process.communicate()
        output = out.decode().splitlines()
        if len(output) == 0:
            err = err.decode()
            print(err)
            return [], [], []
        hash_to_string_dict = {}
        result = []
        method_names = []
        for i, line in enumerate(output):
            parts = line.rstrip().split(' ')
            method_name = parts[0]
            method_names.append(''.join(method_name.split('|')))
            current_result_line_parts = [method_name]
            contexts = parts[1:]
            for context in contexts[:1000]:
                context_ = dict()
                context_parts = context.split(',')
                context_path = context_parts[1]
                context_['name1'] = context_parts[0]
                context_['path'] = context_parts[1]
                context_['name2'] = context_parts[2]
                context_['shortPath'] = context_parts[1]
                pc_info = PathContextInformation(context_)
                current_result_line_parts += [str(pc_info)]
                hash_to_string_dict[(pc_info.token1, pc_info.shortPath, pc_info.token2)] = pc_info
            space_padding = ' ' * (self.config.DATA_NUM_CONTEXTS - len(contexts))
            result_line = ' '.join(current_result_line_parts) + space_padding
            result.append(result_line)
        return result, hash_to_string_dict, method_names
