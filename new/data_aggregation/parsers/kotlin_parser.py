import re
from typing import Tuple, Dict


class KotlinCodeParser:
    @staticmethod
    def _parse_method_name(line: str) -> str:
        pattern = " [^\(]+"
        line = line.strip()
        method_name = re.search(pattern, line).group(0).split(" ")[-1]
        return method_name

    @staticmethod
    def parse_bounds(source_code: str, obj_type: str) -> Dict[str, Tuple[int, int]]:
        # Object types: "method".
        # Returns object_name -> (start_position, end_position).

        name_to_bounds = {}
        name_to_balance = {}  # Number of opened_brackets - closed_brackets.

        source_code_lines = source_code.split("\n")
        for i, line in enumerate(source_code_lines):
            # Found line fun.
            method_name = None

            # To catch such situations:
            # fun getOrCompute(lazyLightClassDataHolder: LazyLightClassDataHolder, diagnostics: () -> Diagnostics) =
            #       cache.computeIfAbsent(lazyLightClassDataHolder, diagnostics)
            ends_with_equals = False

            if "fun" in line:
                method_name = KotlinCodeParser._parse_method_name(line)
                name_to_bounds[method_name] = (i, None)
                name_to_balance[method_name] = 0

                if line.strip().endswith("="):
                    ends_with_equals = True

            opened_counts = line.count("{")
            closed_counts = line.count("}")

            balanced_names = set()  # names for which open_brackets - closed_brackets == 0.
            for name in name_to_balance:
                name_to_balance[name] += (opened_counts - closed_counts)
                if name_to_balance[name] == 0:
                    balanced_names.add(name)

            if ends_with_equals and (method_name is not None) and (method_name in balanced_names):
                balanced_names.remove(method_name)

            for name in balanced_names:
                (start_pos, _) = name_to_bounds[name]
                name_to_bounds[name] = (start_pos, i + 1)
                del name_to_balance[name]

        return name_to_bounds


if __name__ == '__main__':
    with open("data/kotlin/IDELightClassGenerationSupport.kt", "r") as file:
        _source_code = file.read()

    _methods_to_bounds = KotlinCodeParser.parse_bounds(_source_code, "method")
    for _name, _bounds in _methods_to_bounds.items():
        print(f"Method name: {_name} | Bounds: {_bounds}")
