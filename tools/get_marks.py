import ast
import re
import sys


def flatten_attr(node):
    """Recursively retrieves the full dotted name from an ast.Attribute or ast.Name node."""
    if isinstance(node, ast.Attribute):
        print(f"{node.value}.{node.attr}")
        return f"{flatten_attr(node.value)}.{node.attr}"
    elif isinstance(node, ast.Name):
        return node.id
    else:
        # Handle cases where the value is not a Name or Attribute (e.g., a function call)
        # You might need to adjust this based on your specific AST structure
        return str(node)


def get_decorators(source_code):
    """Parses source code and extracts decorator names from function definitions."""
    tree = ast.parse(source_code)
    decorators_info = {}

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
            if node.decorator_list:
                decorators_info[node.name] = []
                for decorator in node.decorator_list:
                    # Decorators can be simple names, attributes (e.g., @manager.register), or calls
                    if isinstance(decorator, ast.Name):
                        decorators_info[node.name].append(decorator.id)
                    elif isinstance(decorator, ast.Attribute):
                        # Handle attribute access like @manager.register
                        # full_name = f"{decorator.value}.{decorator.attr}"
                        v = flatten_attr(decorator.value)
                        full_name = f"{v}"
                        decorators_info[node.name].append(full_name)
                    elif isinstance(decorator, ast.Call):
                        # Handle decorators with arguments like @deco(arg)
                        if isinstance(decorator.func, ast.Name):
                            decorators_info[node.name].append(
                                f"{decorator.func.id}(...)"
                            )
                        elif isinstance(decorator.func, ast.Attribute):
                            full_name = (
                                f"{decorator.func.value}.{decorator.func.attr}(...)"
                            )
                            decorators_info[node.name].append(full_name)

    return decorators_info


code = ""
with open(sys.argv[1], "r") as f:
    code = f.read()


decorator_pattern = r"@pytest.mark.*"
decorators = re.findall(decorator_pattern, code)

# Print the found decorators
for dec in decorators:
    print(dec.strip())


# decorators = get_decorators(code)
# print(decorators)
