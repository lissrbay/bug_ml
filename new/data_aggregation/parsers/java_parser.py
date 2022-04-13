from typing import Tuple, Dict

import javalang


class JavaCodeParser:
    _patterns = {
        "method": javalang.tree.MethodDeclaration,
        "class": javalang.tree.ClassDeclaration
    }

    @staticmethod
    def _parse_bounds(tree, target_node, file_len) -> Tuple[int, int]:
        # source: https://github.com/c2nes/javalang/issues/49
        start = end = None
        for path, node in tree:
            if (start is not None) and (target_node not in path):
                end = node.position.line if node.position is not None else None
                break

            if (start is None) and (node == target_node):
                start = node.position.line if node.position is not None else None

        return start, end if end is not None else file_len  # maybe end - 1?

    @staticmethod
    def parse_bounds(source_code: str, obj_type: str) -> Dict[str, Tuple[int, int]]:
        # Object types: "method", "class".
        # Returns object_name -> (start_position, end_position).

        name_to_bounds = {}
        file_len = len(source_code.split("\n"))
        tree = javalang.parse.parse(source_code)

        for _, node in tree.filter(JavaCodeParser._patterns[obj_type]):
            name_to_bounds[node.name] = JavaCodeParser._parse_bounds(tree, node, file_len)

        return name_to_bounds


if __name__ == '__main__':
    with open("./data/Token.java", "r") as file:
        _source_code = file.read()

    _methods_to_bounds = JavaCodeParser.parse_bounds(_source_code, "method")
    for _name, _bounds in _methods_to_bounds.items():
        print(f"Method name: {_name} | Bounds: {_bounds}")

    _classes_to_bounds = JavaCodeParser.parse_bounds(_source_code, "class")
    for _name, _bounds in _classes_to_bounds.items():
        print(f"Class name: {_name} | Bounds: {_bounds}")
