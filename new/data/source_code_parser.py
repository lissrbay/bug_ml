from typing import Tuple, Dict

import javalang


class JavaCodeParser:
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
    def parse_methods_bounds(source_code: str) -> Dict[str, Tuple[int, int]]:
        method_to_bounds = {}

        file_len = len(source_code.split("\n"))
        tree = javalang.parse.parse(source_code)
        for _, method_node in tree.filter(javalang.tree.MethodDeclaration):
            method_to_bounds[method_node.name] = JavaCodeParser._parse_bounds(tree, method_node, file_len)

        return method_to_bounds

    @staticmethod
    def parse_classes_bounds(source_code: str) -> Dict[str, Tuple[int, int]]:
        class_to_bounds = {}

        file_len = len(source_code.split("\n"))
        tree = javalang.parse.parse(source_code)
        for _, class_node in tree.filter(javalang.tree.ClassDeclaration):
            class_to_bounds[class_node.name] = JavaCodeParser._parse_bounds(tree, class_node, file_len)

        return class_to_bounds


if __name__ == '__main__':
    with open("source_code_examples/Token.java") as file:
        source_code = file.read()

    methods_to_bounds = JavaCodeParser.parse_methods_bounds(source_code)
    for name, bounds in methods_to_bounds.items():
        print(f"Method name: {name:<30} | Bounds: {bounds}")
