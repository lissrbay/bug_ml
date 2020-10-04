import os.path
import re
import sys


class AST:
    def __init__(self, label='', children = [], value = ''):
        self.label = None
        self.children = children
        self.value = value
        self.bounds = ()
        self.type = ''


    def __repr__(self):
        s = 'Tree(label = ' + self.label + ', bounds=' + str(self.bounds) + ', type=' + self.type + ', '
        for child in self.children:
            s += repr(child) + ', '
        s += ")"
        return s


    def get_method_names_and_bounds(self):
        paths = self.paths()
        method_names_and_bounds = []
        for path in paths:
            full_name = ': '.join([i[0] for i in path])
            bounds = path[-1][1]
            method_names_and_bounds.append((full_name, bounds))
        return method_names_and_bounds


    def paths(self, node=None, path=None):
        if node == None:
            node = self
        paths = []
        if path is None:
            path = []
        if node.type != 'code':
            path.append((node.label, node.bounds))
        if node.children:
            for child in node.children:
                paths.extend(self.paths(child, path[:]))
        else:
            paths.append(path)
        return paths


class JavaParser:
    pattern_method_name = re.compile('(?:(?:public|private|protected|static|final|native|synchronized|abstract|transient)+\s+)+[$_\w<>\[\]\s]*\s+[\$_\w]+\([^\)]*\)?\s*?')
    pattern_constructor_name = re.compile("(?:public|protected|private|static)\s*\w+[\(.*?\)]+\s")
    pattern_class = re.compile("(?:public|protected|private|static)\s+(?:class|interface)\s+\w+\s")
    pattern_static = re.compile("(static)\s+\{")
    declaration_patterns = [pattern_method_name,
                            pattern_constructor_name,
                            pattern_class,
                            pattern_static]


    def __init__(self):
        self.brackets_positions = []
        self.labels = []
        self.declaration_types = ['method', 'constructor', 'class', 'static_init']
        self.brackets_positions = []


    def parse(self, txt):
        txt = self.remove_tabs(txt)

        self.brackets_positions.clear()
        self.labels.clear()
        self.brackets_positions.append((-1, 'start'))

        self.recursive_parsing(txt)
        self.find_declarations(txt)
        self.fill_spaces()

        ast = self.construct_ast(curr_position = 0)
        ast = ast.children[0]
        return ast


    def create_node(self, label=('', '')):
        root = AST(children=[])
        root.label = label[0]
        root.type = label[1]
        return root


    def construct_ast(self, label=('', ''), pos = 0, curr_position=0):
        root = self.create_node(label)

        for i, val in enumerate(self.brackets_positions[1:]):
            if i < curr_position:
                continue
            pos_end, bracket = val
            if bracket == '{':
                child_label = self.labels[curr_position]
                curr_position += 1
                if child_label[1] in self.declaration_types:
                    child, curr_position = self.construct_ast(child_label, pos_end, curr_position)
                else:
                    child, curr_position = self.construct_ast(('code', 'code'), pos_end, curr_position)
                root.children.append(child)
            else:
                curr_position += 1
                root.bounds = (pos, pos_end)
                return root, curr_position

        return root


    def recursive_parsing(self, txt, pos=0):
        next_pos = 0
        for i, char in enumerate(txt[pos:], pos):
            if i <= next_pos:
                continue
            if char == '{':
                self.brackets_positions.append((i, '{'))
                pos = i + 1
                next_pos = self.recursive_parsing(txt, pos)
            if char == '}':
                self.brackets_positions.append((i, '}'))
                return i


    def fill_spaces(self):
        j = 0
        for i in range(1,len(self.brackets_positions)):
            if j < len(self.labels) and self.labels[j][2] <= self.brackets_positions[i][0]:
                j += 1
                continue
            self.labels.insert(j, ('code', 'code', self.brackets_positions[0]))
            j += 1


    def find_declarations(self, code):
        all_declarations = []
        for declaration_pattern, type in zip(self.declaration_patterns, self.declaration_types):
            declarations = self.find_declarations_by_pattern(declaration_pattern, code, type)
            if declarations:
                if type == 'static_init':
                    declarations = [('static', type, info[2]-(len(info[0])-6)) for info in declarations]
                all_declarations.extend(declarations)

        all_declarations.sort(key=lambda x: x[2])
        self.labels = all_declarations


    def find_declarations_by_pattern(self, pattern, code, type):
        declarations = [(m.group(0), type, m.end(0)) for m in re.finditer(pattern, code)]
        return declarations


    def remove_tabs(self, code):
        code = ''.join(code)
        code = re.sub(' +', ' ', code)
        return re.sub('\t+', '', code)


if __name__ == "__main__":
    jp = JavaParser()
    f = open("C:\\Users\\lissrbay\\Desktop\\bugml\\uni_proj\\src\\com\\nyancake\\pack_2\\FNV.java", 'r')
    print(jp.parse(f.readlines()))