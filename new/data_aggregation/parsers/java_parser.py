from typing import Tuple, Dict

import javalang


class JavaCodeParser:
    def __init__(self):
        self._patterns = {
            "method": javalang.tree.MethodDeclaration,
            "class": javalang.tree.ClassDeclaration
        }

    def _parse_bounds(self, tree, target_node, file_len) -> Tuple[int, int]:
        # source: https://github.com/c2nes/javalang/issues/49
        start = end = None
        for path, node in tree:
            if (start is not None) and (target_node not in path):
                end = node.position.line - 1 if node.position is not None else None
                break

            if (start is None) and (node == target_node):
                start = node.position.line - 1 if node.position is not None else None

        return start, end - 1 if end is not None else file_len  # maybe end - 1?

    def parse_bounds(self, source_code: str, obj_type: str) -> Dict[str, Tuple[int, int]]:
        # Object types: "method", "class".
        # Returns object_name -> (start_position, end_position).

        name_to_bounds = {}
        file_len = len(source_code.split("\n"))
        tree = javalang.parse.parse(source_code)

        for _, node in tree.filter(self._patterns[obj_type]):
            name_to_bounds[node.name] = self._parse_bounds(tree, node, file_len)

        return name_to_bounds

    def parse_body(self, source_code: str, obj_type: str) -> Dict[str, str]:
        name_to_bounds = self.parse_bounds(source_code, obj_type)
        source_code_lines = source_code.split("\n")
        name_to_body = {}

        for name, (start_pos, end_pos) in name_to_bounds.items():
            body = "\n".join(source_code_lines[start_pos:end_pos])
            name_to_body[name] = body

        return name_to_body


if __name__ == '__main__':
    with open("./data/java/Token.java", "r") as file:
        _source_code = file.read()

    parser = JavaCodeParser()

    _methods_to_bounds = parser.parse_bounds(_source_code, "method")
    for _name, _bounds in _methods_to_bounds.items():
        print(f"Method name: {_name} | Bounds: {_bounds}")

    print()

    _classes_to_bounds = parser.parse_bounds(_source_code, "class")
    for _name, _bounds in _classes_to_bounds.items():
        print(f"Class name: {_name} | Bounds: {_bounds}")

    print()

    _methods_to_body = parser.parse_body(_source_code, "method")
    for _name, _body in _methods_to_body.items():
        print(f"Method name: {_name}")
        print(f"Body:")
        print(_body)
        print()
