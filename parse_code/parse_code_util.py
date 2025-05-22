import ast
import random
import re
import ast
from collections import defaultdict
from extract_cpp_locations import record_identifier_occurrences
import tempfile
from typing import Dict, Any # Use Any if the return type of the helper is unknown
from extract_js_locations import parse_js
from extract_go_locations import parse_go
from extract_java_locations import parse_java

def remove_comments(code: str) -> str:
    """
    Removes Python comments (#...) and docstrings that appear as standalone
    blocks (i.e., triple-quoted strings preceded only by whitespace).
    
    Preserves triple-quoted strings if they appear in expressions or
    assignments (i.e., on the same line with non-whitespace characters
    before the triple quotes).

    :param code: A string containing Python code.
    :return: Cleaned code as a string.
    """
    lines = code.splitlines(True)  # keep line endings
    result = []
    
    in_docstring_block = False
    docstring_delimiter = None

    for line in lines:
        # If we are currently skipping a standalone docstring block:
        if in_docstring_block:
            # Check if this line ends the docstring block
            end_index = line.find(docstring_delimiter)
            if end_index == -1:
                # Entire line is still inside the docstring block, skip it
                continue
            else:
                # We found the closing triple quotes
                # Everything up to (and including) those quotes is removed,
                # but there might be code after them on the same line.
                after_quotes = line[end_index + len(docstring_delimiter):]
                in_docstring_block = False
                docstring_delimiter = None

                # Now handle anything after the docstring on the same line
                # Remove inline comments if any
                comment_pos = after_quotes.find('#')
                if comment_pos != -1:
                    after_quotes = after_quotes[:comment_pos]
                
                # Keep remainder if it’s not blank
                if after_quotes.strip():
                    result.append(after_quotes.rstrip() + '\n')
            continue
        
        # --- Not currently in a standalone docstring block ---
        
        # Strip left side to check indentation vs. docstring usage
        lstrip_line = line.lstrip()
        
        # Check if this line starts with a triple-quoted string (standalone docstring)
        # i.e., no other characters before the triple quotes
        if (lstrip_line.startswith('"""') or lstrip_line.startswith("'''")) \
           and (len(line) - len(lstrip_line) == 0 or  # no indentation
                line[: -len(lstrip_line)].isspace()):  # only indentation
            # This is a standalone docstring start
            # Figure out which delimiter was used
            docstring_delimiter = '"""' if lstrip_line.startswith('"""') else "'''"
            
            # Check if it ends on the same line
            rest = lstrip_line[len(docstring_delimiter):]
            closing_pos = rest.find(docstring_delimiter)
            if closing_pos == -1:
                # Docstring continues on subsequent lines
                in_docstring_block = True
            else:
                # Docstring opens and closes on the same line
                after_quotes = rest[closing_pos + len(docstring_delimiter):]
                # Remove any inline comment after the docstring
                comment_pos = after_quotes.find('#')
                if comment_pos != -1:
                    after_quotes = after_quotes[:comment_pos]
                
                if after_quotes.strip():
                    # keep any leftover code
                    # restore the original indentation
                    indent_size = len(line) - len(lstrip_line)
                    leftover = ' ' * indent_size + after_quotes
                    result.append(leftover.rstrip() + '\n')
            continue
        
        # If line starts with # (plus optional indentation), skip it entirely
        if lstrip_line.startswith('#'):
            continue
        
        # Otherwise, remove inline comment if any
        comment_pos = line.find('#')
        if comment_pos != -1:
            # Keep everything up to the #, strip trailing spaces
            line = line[:comment_pos].rstrip() + '\n'

        # Append the resulting line if not empty
        if line.strip():
            result.append(line)

    return "".join(result)

def find_variables_and_functions_with_occurrences_js(code_text):
    return parse_js(code_text)


def find_variables_and_functions_with_occurrences_cpp(code_text):
    """
    Finds C++ identifiers and their occurrences using a temporary file.

    Writes the input code string to a secure temporary file, calls a
    processing function that requires a filename, and ensures the
    temporary file is cleaned up afterwards.

    Args:
        code_text: A string containing the C++ source code.

    Returns:
        A dictionary mapping identifier names (str) to their counts (int),
        as returned by record_identifier_occurrences. Returns an empty
        dictionary if input is empty or an error occurs.
    """
    if not code_text:
        print("Input code text is empty.")
        return {}

    func_and_var_names = {}
    try:
        # Create a named temporary file that is automatically deleted
        # Use 'w' for write mode. Specify encoding for consistency.
        # delete=True (default) ensures cleanup.
        # suffix is optional but can be helpful for tools that check extensions.
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.cpp', delete=True) as temp_f:
            # Write the code text to the temporary file
            temp_f.write(code_text)

            # IMPORTANT: Flush the buffer to ensure the data is written to disk
            # before the record_identifier_occurrences function tries to read it.
            temp_f.flush()

            # Get the path/name of the temporary file
            temp_file_path = temp_f.name
            # print(f"Created temporary file: {temp_file_path}") # For debugging

            # Call the function that requires a file path
            func_and_var_names = record_identifier_occurrences(temp_file_path)

            # The file will be closed and deleted automatically when exiting the 'with' block

    except IOError as e:
        print(f"Error creating or writing to temporary file: {e}")
        return {'func_names': {}, 'variables': {}} # Return empty dict on I/O error
    except Exception as e:
        # Catch potential errors from record_identifier_occurrences
        print(f"An unexpected error occurred during processing: {e}")
        return {'func_names': {}, 'variables': {}} # Return empty dict on other errors

    return func_and_var_names
    


def find_variables_and_functions_with_occurrences_python(code: str):
    """
    Parses the given Python code and returns a dictionary with:
        {
            "variables": {
                var_name: [
                    {"loc": (start_offset, end_offset), "text": substring_from_code},
                    ...
                ],
                ...
            },
            "func_names": {
                func_name: [
                    {"loc": (start_offset, end_offset), "text": substring_from_code},
                    ...
                ],
                ...
            }
        }

    - 'variables' includes:
        - Names that are assigned (Store context).
        - Function parameters.
    - 'func_names' includes names of function/async function definitions.
    - Each occurrence dict includes:
        {
          "loc": (start_offset, end_offset),
          "text": substring_from_code
        }
      extracted directly from the original code string.
    """

    # Use keepends=True so that newlines are preserved in each line.
    lines = code.splitlines(keepends=True)

    variable_names = set()
    function_names = set()
    all_name_occurrences = defaultdict(list)

    # A helper to find (approximate) column offset for the function name.
    def get_function_name_col_offset(line_str, start_index=0):
        i = start_index

        # Skip leading whitespace
        while i < len(line_str) and line_str[i].isspace():
            i += 1

        # If we see 'async', skip it plus the following space.
        if line_str[i:].startswith("async"):
            i += len("async")
            while i < len(line_str) and line_str[i].isspace():
                i += 1

        # Now skip 'def'
        if line_str[i:].startswith("def"):
            i += len("def")
            while i < len(line_str) and line_str[i].isspace():
                i += 1

        return i

    class Collector(ast.NodeVisitor):
        def visit_Name(self, node):
            name = node.id
            lineno = node.lineno
            col_offset = node.col_offset

            # Record the raw occurrence
            all_name_occurrences[name].append((lineno, col_offset))

            # If it's a Store context, treat it as a defined variable.
            if isinstance(node.ctx, ast.Store):
                variable_names.add(name)

            self.generic_visit(node)

        def visit_FunctionDef(self, node):
            # Record function name.
            func_name = node.name
            function_names.add(func_name)

            # Use an approximate offset for the function name rather than for 'def'
            line_index = node.lineno - 1
            if 0 <= line_index < len(lines):
                line_str = lines[line_index]
                name_col = get_function_name_col_offset(line_str, node.col_offset)
            else:
                name_col = node.col_offset

            # Save function name occurrence.
            all_name_occurrences[func_name].append((node.lineno, name_col))

            # Visit the function body (so we see local definitions, etc.)
            self.generic_visit(node)

        def visit_AsyncFunctionDef(self, node):
            func_name = node.name
            function_names.add(func_name)

            line_index = node.lineno - 1
            if 0 <= line_index < len(lines):
                line_str = lines[line_index]
                name_col = get_function_name_col_offset(line_str, node.col_offset)
            else:
                name_col = node.col_offset

            all_name_occurrences[func_name].append((node.lineno, name_col))

            self.generic_visit(node)

        def visit_arg(self, node):
            """
            Capture function parameter names as variables.
            """
            param_name = node.arg
            variable_names.add(param_name)

            # If lineno/col_offset exist, record them.
            lineno = getattr(node, 'lineno', None)
            col_offset = getattr(node, 'col_offset', None)
            if lineno is not None and col_offset is not None:
                all_name_occurrences[param_name].append((lineno, col_offset))

            self.generic_visit(node)

    # Parse and walk the AST.
    tree = ast.parse(code)
    Collector().visit(tree)

    result = {
        "variables": {},
        "func_names": {}
    }

    # Helper function to compute the absolute offset in the code string given a line number and column offset.
    def compute_absolute_offset(line_no, col_offset):
        # line_no is 1-indexed; sum the lengths of all previous lines.
        return sum(len(lines[i]) for i in range(line_no - 1)) + col_offset

    def get_occurrence_info(name, line_no, col_off):
        start_offset = compute_absolute_offset(line_no, col_off)
        end_offset = start_offset + len(name)
        # Extract snippet from the original code.
        snippet = code[start_offset:end_offset]
        return {"loc": (start_offset, end_offset), "text": snippet}

    # Collect all variable occurrences.
    for var in variable_names:
        occurrences = []
        for ln, co in all_name_occurrences[var]:
            info = get_occurrence_info(var, ln, co)
            occurrences.append(info)
        result["variables"][var] = occurrences

    # Collect all function name occurrences.
    for fn in function_names:
        occurrences = []
        for ln, co in all_name_occurrences[fn]:
            info = get_occurrence_info(fn, ln, co)
            occurrences.append(info)
        result["func_names"][fn] = occurrences

    return result


def find_variables_and_functions_with_occurrences_go(code_text):
    return parse_go(code_text)

def find_variables_and_functions_with_occurrences_java(code_text):
    return parse_java(code_text)


import keyword

def is_valid_variable_name(name):
    """
    Checks if the given string can be a valid Python variable name.

    Args:
        name (str): The string to check.

    Returns:
        bool: True if the string is a valid variable name, False otherwise.
    """
    return name.isidentifier() and not keyword.iskeyword(name)