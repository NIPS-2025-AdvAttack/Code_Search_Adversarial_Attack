#!/usr/bin/env python3
import argparse
import clang.cindex
import os
import json
import collections # Added for defaultdict
clang.cindex.Config.set_library_file("/usr/lib/llvm-15/lib/libclang.so")
# If needed, set the libclang path:
# clang.cindex.Config.set_library_file('/path/to/libclang.so')

import re
import unittest
from typing import List

def get_variable_context_regex(input_string: str) -> str:
    """
    Generates a regex to find an input string when it's not immediately
    preceded or followed by characters typically allowed within a C/C++
    variable name (alphanumeric or underscore).

    Args:
        input_string: The literal string to search for.

    Returns:
        A raw regex string pattern.

    Raises:
        TypeError: If input_string is not a string.
        ValueError: If input_string is empty.
    """
    if not isinstance(input_string, str):
        raise TypeError("Input must be a string.")
    if not input_string:
        raise ValueError("Input string cannot be empty.")

    escaped_input = re.escape(input_string)
    # Character class for typical C/C++ variable name characters
    var_char_class = r"[a-zA-Z0-9_]"
    # Regex using negative lookbehind and lookahead
    regex_pattern = rf"(?<!{var_char_class}){escaped_input}(?!{var_char_class})"
    return regex_pattern

def find_var_outside_quotes(target_string: str, text: str) -> List[re.Match]:
    """
    Finds occurrences of target_string in text that meet variable context
    rules (using get_variable_context_regex) AND are not inside double
    quotes ("..."). Handles escaped quotes (\").

    Args:
        target_string: The literal string (variable name) to search for.
        text: The text content to search within.

    Returns:
        A list of re.Match objects for valid occurrences found outside
        double quotes.

    Raises:
        TypeError: If target_string or text are not strings.
        ValueError: If target_string is empty.
    """
    # Input validation
    if not isinstance(target_string, str):
        raise TypeError("Target string must be a string.")
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not target_string:
        raise ValueError("Target string cannot be empty.")

    # Get the base regex ensuring variable context
    try:
        base_regex = get_variable_context_regex(target_string)
    except (TypeError, ValueError) as e:
        # Propagate errors from regex generation
        raise e

    valid_matches: List[re.Match] = []

    # Find all potential matches based on variable context first
    for match in re.finditer(base_regex, text):
        start_index = match.start()
        preceding_text = text[:start_index]

        # Count unescaped double quotes before the match's start position
        # to determine if we are currently inside quotes.
        quote_count = 0
        i = 0
        while i < len(preceding_text):
            char = preceding_text[i]
            if char == '"':
                # Found an unescaped double quote
                quote_count += 1
            elif char == '\\':
                # Found an escape character. Skip the *next* character,
                # as it's escaped (e.g., \", \\, \n etc.)
                # Ensure we don't go out of bounds if \ is the last char
                if i + 1 < len(preceding_text):
                    i += 1
                # else: A trailing backslash - treat normally or raise error?
                # Current behavior: allows it, next loop iteration handles normally.
            i += 1

        # If the number of unescaped double quotes is even,
        # the match starts outside of a quoted section.
        if quote_count % 2 == 0:
            valid_matches.append(match)

    return valid_matches

def replace_var_outside_quotes(target_string: str, replacement_string: str, text: str) -> str:
    """
    Replaces occurrences of target_string with replacement_string in text,
    but only when they meet variable context rules AND are not inside
    double quotes (as determined by find_var_outside_quotes).

    Args:
        target_string: The string (variable name) to be replaced.
        replacement_string: The string to replace with.
        text: The text content to perform replacements on.

    Returns:
        A new string with the valid replacements made.

    Raises:
        TypeError: If target_string, replacement_string, or text are not strings.
        ValueError: If target_string is empty.
    """
    # Input validation
    if not isinstance(target_string, str):
        raise TypeError("Target string must be a string.")
    if not isinstance(replacement_string, str):
        raise TypeError("Replacement string must be a string.")
    if not isinstance(text, str):
        raise TypeError("Text must be a string.")
    if not target_string:
        # Or return text depending on desired behavior for empty target
        raise ValueError("Target string cannot be empty.")


    # Find all valid matches first
    try:
        matches = find_var_outside_quotes(target_string, text)
    except (TypeError, ValueError) as e:
        # Propagate errors from finding matches
        raise e

    # If no valid matches were found, return the original text unmodified
    if not matches:
        return text

    # Build the result string piece by piece, handling index shifts
    result_parts = []
    last_end = 0
    
    for match in matches:
        start_pos = match.start()
        end_pos = match.end()
        
        # Append the text segment before the current match
        result_parts.append(text[last_end:start_pos])
        # Append the replacement string
        result_parts.append(replacement_string)
        # Update the position for the next segment
        last_end = end_pos

    # Append the remaining part of the text after the last match
    result_parts.append(text[last_end:])
    
    return "".join(result_parts)

BOOST_INCLUDE_PATH = "/data/usr/codesearch/boost_installation/boost_1_74_0_installed_targets/include/" # Adjust if necessary

def parse_source(file_path, extra_args=None):
    """Parse the C/C++ source file and return its translation unit."""
    index = clang.cindex.Index.create()
    if extra_args is None:
        extra_args = ["-std=c++11"]
    # Enable detailed preprocessing records so macro instantiations are available.

    extra_args += ["-Xclang", "-detailed-preprocessing-record"]
    # Example include path, adjust as needed
    # extra_args += [f"-I {BOOST_INCLUDE_PATH}"]
    # Add standard C++ include paths if necessary for better parsing
    # extra_args += ['-std=c++11', '-I/usr/include/c++/11', '-I/usr/include/x86_64-linux-gnu/c++/11'] # Example for g++ 11 on Linux
    options = (
        clang.cindex.TranslationUnit.PARSE_INCOMPLETE |
        clang.cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD
    )
    try:
        tu = index.parse(file_path, args=extra_args, options=options)
        if not tu:
            print(f"Error: Failed to parse {file_path}. TranslationUnit is None.")
            return None
            # return None
        return tu
    except clang.cindex.LibclangError as e:
        print(f"Error: Libclang error parsing {file_path}: {e}")
        return None
    except Exception as e:
        print(f"Error: Unexpected error during parsing {file_path}: {e}")
        return None


def get_function_extents(cursor, file_path):
    """
    Recursively find all function definitions in the current file and return a list of their extents.
    Each extent is a tuple of (start_offset, end_offset) in bytes.
    """
    extents = []
    abs_file_path = os.path.abspath(file_path)
    for child in cursor.get_children():
        try:
            # Ensure we only process nodes with valid locations and files
            if not child.location or not child.location.file:
                continue

            # Check if the node is in the target file
            if os.path.abspath(child.location.file.name) == abs_file_path:
                if ((child.kind == clang.cindex.CursorKind.FUNCTION_DECL or
                     child.kind == clang.cindex.CursorKind.CXX_METHOD or
                     child.kind == clang.cindex.CursorKind.CONSTRUCTOR) and child.is_definition()):
                    extents.append((child.extent.start.offset, child.extent.end.offset))

            # Recurse into children regardless of file, but only add extents if they are in the target file
            extents.extend(get_function_extents(child, file_path))
        except Exception as e:
            # print(f"Warning: Error processing child node {child.spelling} ({child.kind}): {e}")
            pass # Continue processing other children
    return extents


def is_from_current_file(cursor, file_path):
    """Return True if the cursor is from the main file."""
    if not cursor or not cursor.location or not cursor.location.file:
        return False
    try:
        return os.path.abspath(cursor.location.file.name) == os.path.abspath(file_path)
    except Exception as e:
        # print(f"Warning: Could not resolve path for cursor {cursor.spelling}: {e}")
        return False

def get_directive_lines(file_path):
    """
    Given a file path, return a set of line numbers (1-indexed)
    that start with '#' (after stripping whitespace).
    Handles potential decoding errors.
    """
    directive_lines = set()
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for i, line in enumerate(lines, start=1):
            if line.strip().startswith("#"):
                directive_lines.add(i)
    except Exception as e:
        print(f"Warning: Could not read or process directives in {file_path}: {e}")
    return directive_lines


def should_skip_token(token, file_path, directive_lines):
    """
    Checks if a token should be skipped based on various criteria.
    """
    # Known external typedefs (common C types)
    external_types = {
        "uint8_t", "uint16_t", "uint32_t", "uint64_t",
        "int8_t", "int16_t", "int32_t", "int64_t",
        "size_t", "ssize_t", "ptrdiff_t", "nullptr_t",
        "bool", "char", "short", "int", "long", "float", "double", "void",
        "wchar_t", "char16_t", "char32_t", "std::string", "std::wstring",
        "std::vector", "std::list", "std::map", "std::set",
    }

    if token.spelling in external_types:
        return True

    # Skip tokens on preprocessor directive lines.
    if token.location.line in directive_lines:
        return True

    # Skip if location is invalid
    if not token.location or not token.location.file:
        return True

    # Skip if not in the main file (already checked mostly, but good safety check)
    if not is_from_current_file(token.cursor, file_path):
         # Exception: Allow macro instantiations even if definition is elsewhere,
         # but only if the instantiation *location* is in the current file.
         if token.kind != clang.cindex.TokenKind.IDENTIFIER or \
            token.cursor.kind != clang.cindex.CursorKind.MACRO_INSTANTIATION or \
            not is_from_current_file(token.cursor, file_path): # Check instantiation location
                if not (token.cursor.referenced and is_from_current_file(token.cursor.referenced, file_path)):
                     return True


    # Check semantic parent for "std" namespace or common library namespaces
    # Be careful with semantic_parent, it can be None
    try:
        if token.cursor and token.cursor.semantic_parent:
            parent_spelling = token.cursor.semantic_parent.spelling
            # Broader check for common system/library namespaces
            if parent_spelling == "std" or parent_spelling.startswith("__"):
                 # Check if the definition is also external
                 if token.cursor.referenced and not is_from_current_file(token.cursor.referenced, file_path):
                    return True
                 # If no referenced cursor or it's internal, double check if it's a definition within std::
                 if parent_spelling == "std" and not token.cursor.is_definition():
                     # Likely a usage of std::something, skip if definition is external
                      if not token.cursor.referenced or not is_from_current_file(token.cursor.referenced, file_path):
                          return True

    except Exception:
        pass # Ignore errors during semantic parent checks


    # If the token refers to an external declaration (and is not a macro instantiation in the current file).
    # Need to be careful not to skip local usages of externally defined types/functions if desired.
    # This check focuses on the *definition* location.
    if token.cursor and token.cursor.referenced:
        if not is_from_current_file(token.cursor.referenced, file_path):
            # Allow using external types/functions, but skip if *this specific token*
            # points directly to the external definition site (less common for usage tokens).
            # Let's refine this: skip only if the *definition* is external.
            # The is_from_current_file check on token.cursor.referenced handles this.
            return True # Skip tokens whose definition is outside the current file

    # Skip built-in function names or identifiers.
    if token.spelling.startswith("__builtin_") or token.spelling.startswith("__"):
         # Check if it's a known C/C++ keyword (Clang might sometimes tokenize keywords as identifiers)
        keywords = {"auto", "break", "case", "char", "const", "continue", "default", "do", "double",
                    "else", "enum", "extern", "float", "for", "goto", "if", "int", "long", "register",
                    "return", "short", "signed", "sizeof", "static", "struct", "switch", "typedef",
                    "union", "unsigned", "void", "volatile", "while", "class", "public", "private",
                    "protected", "new", "delete", "this", "throw", "try", "catch", "namespace", "using",
                    "true", "false", "nullptr", "constexpr", "static_assert", "decltype", "endl", "cout"}
        if token.spelling in keywords:
            return True
        # If it starts with __ and refers to something external, skip it.
        if token.cursor and token.cursor.referenced and not is_from_current_file(token.cursor.referenced, file_path):
           return True


    return False

import io
def remove_cpp_comments(code_string):
  """
  Removes C/C++ style comments (// and /* */) from a code string.

  Handles comments within strings and character literals correctly.

  Args:
    code_string: A string containing C/C++ code.

  Returns:
    A string with all comments removed.
  """
  # Use io.StringIO for efficient string building
  result = io.StringIO()
  n = len(code_string)
  i = 0
  in_multiline_comment = False
  in_singleline_comment = False
  in_string = False
  in_char = False

  while i < n:
    # --- Handle Exiting Comments/Strings/Chars ---
    if in_multiline_comment:
      if code_string[i:i+2] == '*/':
        in_multiline_comment = False
        i += 2 # Skip '*/'
      else:
        i += 1 # Skip character inside comment
      continue # Move to next iteration

    if in_singleline_comment:
      if code_string[i] == '\n':
        in_singleline_comment = False
        result.write('\n') # Keep the newline
        i += 1
      else:
        i += 1 # Skip character inside comment
      continue # Move to next iteration

    if in_string:
      result.write(code_string[i])
      if code_string[i] == '\\' and i + 1 < n: # Handle escaped quote
          result.write(code_string[i+1])
          i += 2
      elif code_string[i] == '"':
          in_string = False
          i += 1
      else:
          i += 1
      continue # Move to next iteration

    if in_char:
      result.write(code_string[i])
      if code_string[i] == '\\' and i + 1 < n: # Handle escaped quote
          result.write(code_string[i+1])
          i += 2
      elif code_string[i] == '\'':
          in_char = False
          i += 1
      else:
          i += 1
      continue # Move to next iteration

    # --- Handle Entering Comments/Strings/Chars (if not already in one) ---
    if code_string[i:i+2] == '/*':
        in_multiline_comment = True
        i += 2 # Skip '/*'
    elif code_string[i:i+2] == '//':
        in_singleline_comment = True
        i += 2 # Skip '//'
    elif code_string[i] == '"':
        in_string = True
        result.write('"')
        i += 1
    elif code_string[i] == '\'':
        in_char = True
        result.write('\'')
        i += 1
    else:
        # Regular code character
        result.write(code_string[i])
        i += 1

  return result.getvalue()

def complete_occurrences_with_string_matches(occurrences, file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
    # content = remove_cpp_comments(content) # Remove comments from the content
    for name, _ in occurrences.items():
        # Find all occurrences of the name 
        loc_list = []
        matches = find_var_outside_quotes(name, content)
        for match in matches:
            start_pos = match.start()
            end_pos = match.end()
            # update the loc_list
            loc_list.append({
                # "file": file_path,
                # "line": "PlaceHolder",  # Placeholder, not accurate
                # "column": "PlaceHolder",  # Placeholder, not accurate
                "loc": (start_pos, end_pos),
                "text": name,
                # "start_offset": start_pos,
                # "end_offset": end_pos,
            })
            # update the occurrences
            occurrences[name] = loc_list
    return occurrences
        
def record_identifier_occurrences(file_path):
    """
    Parses the C/C++ file and records occurrences of various identifier types.
    Returns a dictionary mapping identifier types to names and their locations.
    """
    abs_file_path = os.path.abspath(file_path)
    tu = parse_source(file_path)
    if not tu:
        # return f"Could not parse file: {file_path}"
        # raise Exception(f"Could not parse file: {file_path}")
        return {"func_names": {}, "variables": {}, "error": f"Could not parse file: {file_path}"}

    tokens = list(tu.get_tokens(extent=tu.cursor.extent))
    function_extents = get_function_extents(tu.cursor, file_path)
    directive_lines = get_directive_lines(file_path)


    # Use defaultdict for easier appending
    var_occurrences = collections.defaultdict(list)
    fun_occurrences = collections.defaultdict(list)
    method_occurrences = collections.defaultdict(list) # Includes constructors
    field_occurrences = collections.defaultdict(list)
    class_occurrences = collections.defaultdict(list) # Includes struct, union, enum declarations
    typedef_occurrences = collections.defaultdict(list)
    macro_occurrences = collections.defaultdict(list)
    enum_const_occurrences = collections.defaultdict(list)
    alias_occurrences = collections.defaultdict(list)
    namespace_occurrences = collections.defaultdict(list)
    label_occurrences = collections.defaultdict(list) # Added for labels (e.g., goto targets)
    param_occurrences = collections.defaultdict(list) # Specifically track parameter declarations
    undeclared_occurrences = collections.defaultdict(list)
    template_param_occurrences = collections.defaultdict(list)
    function_template_occurrences = collections.defaultdict(list)
    class_template_occurrences = collections.defaultdict(list)

    # Define AST cursor kinds for classification
    # Note: DECL_REF_EXPR and MEMBER_REF_EXPR need checking referenced kind
    var_kinds = {clang.cindex.CursorKind.VAR_DECL}
    param_kinds = {clang.cindex.CursorKind.PARM_DECL}
    fun_kinds = {clang.cindex.CursorKind.FUNCTION_DECL, clang.cindex.CursorKind.CALL_EXPR}
    method_kinds = {clang.cindex.CursorKind.CXX_METHOD, clang.cindex.CursorKind.CONSTRUCTOR, clang.cindex.CursorKind.DESTRUCTOR}
    field_kinds = {clang.cindex.CursorKind.FIELD_DECL}
    # Broadened class_kinds to include enum declarations as well
    class_kinds = {clang.cindex.CursorKind.CLASS_DECL, clang.cindex.CursorKind.STRUCT_DECL, clang.cindex.CursorKind.UNION_DECL, clang.cindex.CursorKind.ENUM_DECL}
    typedef_kinds = {clang.cindex.CursorKind.TYPEDEF_DECL}
    alias_kinds = {clang.cindex.CursorKind.TYPE_ALIAS_DECL}
    enum_const_kinds = {clang.cindex.CursorKind.ENUM_CONSTANT_DECL}
    macro_def_kinds = {clang.cindex.CursorKind.MACRO_DEFINITION}
    macro_inst_kinds = {clang.cindex.CursorKind.MACRO_INSTANTIATION}
    namespace_kinds = {clang.cindex.CursorKind.NAMESPACE, clang.cindex.CursorKind.NAMESPACE_REF}
    label_kinds = {clang.cindex.CursorKind.LABEL_STMT, clang.cindex.CursorKind.LABEL_REF}
    type_ref_kinds = {clang.cindex.CursorKind.TYPE_REF} # Reference to a type
    member_ref_kinds = {clang.cindex.CursorKind.MEMBER_REF, clang.cindex.CursorKind.MEMBER_REF_EXPR} # Reference to field or method
    decl_ref_kinds = {clang.cindex.CursorKind.DECL_REF_EXPR} # Reference to var, func, enum const etc.
    template_param_kinds = {
        clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER,
        clang.cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER, # e.g., template<int N>
        clang.cindex.CursorKind.TEMPLATE_TEMPLATE_PARAMETER, # e.g., template<template<typename> class C>
    }
    function_template_kinds = {
        clang.cindex.CursorKind.FUNCTION_TEMPLATE,
        # clang.cindex.CursorKind.CXX_METHOD_TEMPLATE,
    }
    class_template_kinds = {
        clang.cindex.CursorKind.CLASS_TEMPLATE,
        clang.cindex.CursorKind.CLASS_TEMPLATE_PARTIAL_SPECIALIZATION,
        # clang.cindex.CursorKind.CLASS_TEMPLATE_SPECIALIZATION,
    }


    processed_tokens = set() # Keep track of (start_offset, end_offset) to avoid duplicates

    for token in tokens:
        token_key = (token.extent.start.offset, token.extent.end.offset)
        if token_key in processed_tokens:
            continue

        # Basic token checks
        if token.kind != clang.cindex.TokenKind.IDENTIFIER:
            continue
        if not token.cursor or token.cursor.kind == clang.cindex.CursorKind.INVALID_FILE:
             continue
        if not token.location or not token.location.file:
            continue

        # Skip based on filtering rules
        if should_skip_token(token, file_path, directive_lines):
            continue

        # Ensure the token's *location* is in the current file for recording
        if os.path.abspath(token.location.file.name) != abs_file_path:
             continue

        # --- Start Classification ---
        cursor = token.cursor
        kind = cursor.kind
        referenced = cursor.referenced # Can be None

        # Determine the most specific kind, resolving references
        effective_kind = kind
        target_cursor = cursor # The cursor representing the identifier's definition/declaration or reference nature

        if kind in decl_ref_kinds and referenced:
            effective_kind = referenced.kind
            target_cursor = referenced
        elif kind in member_ref_kinds and referenced:
            effective_kind = referenced.kind
            target_cursor = referenced
        elif kind in type_ref_kinds and referenced:
             # TYPE_REF's referenced cursor points to the type declaration (class, struct, typedef, etc.)
             effective_kind = referenced.kind
             target_cursor = referenced


        # Get occurrence details
        orig_name = token.spelling
        occurrence_details = {
            "file": abs_file_path, # Always store absolute path
            "line": token.location.line,
            "column": token.location.column,
            "start_offset": token.extent.start.offset,
            "end_offset": token.extent.end.offset,
            # Add kind information for context
            "token_kind": token.kind.name,
            "cursor_kind": kind.name,
            "effective_kind": effective_kind.name,
            # "is_definition": target_cursor.is_definition() if target_cursor else False,
        }

        # --- Categorize based on effective_kind ---
        if orig_name == "std":
            namespace_occurrences[orig_name].append(occurrence_details)
            processed_tokens.add(token_key)
            continue
        # Handle Macros first (as they might overlap with other kinds)
        if kind in macro_inst_kinds:
             # Make sure the instantiation itself is in the current file
            if is_from_current_file(cursor, file_path):
                macro_occurrences[orig_name].append(occurrence_details)
                processed_tokens.add(token_key)
                continue # Processed as macro, skip other checks
        # Also record macro definitions defined *in* this file
        elif kind in macro_def_kinds:
             if is_from_current_file(cursor, file_path):
                macro_occurrences[orig_name].append(occurrence_details)
                processed_tokens.add(token_key)
                continue

        # Parameter Declarations
        if effective_kind in param_kinds:
            param_occurrences[orig_name].append(occurrence_details)
        # Function/Method Declarations/Calls
        elif effective_kind in fun_kinds or effective_kind in method_kinds:
             # Distinguish methods/constructors from standalone functions
            if effective_kind in method_kinds:
                 method_occurrences[orig_name].append(occurrence_details)
            else: # Regular functions
                 fun_occurrences[orig_name].append(occurrence_details)
        # Class/Struct/Union/Enum Declarations or References
        elif effective_kind in class_kinds:
            class_occurrences[orig_name].append(occurrence_details)
        # Field (Member Variable) Declarations or References
        elif effective_kind in field_kinds:
            field_occurrences[orig_name].append(occurrence_details)
        # Variable Declarations or References (excluding parameters and fields)
        elif effective_kind in var_kinds:
            var_occurrences[orig_name].append(occurrence_details)
        # Typedef Declarations or References
        elif effective_kind in typedef_kinds:
            typedef_occurrences[orig_name].append(occurrence_details)
        # Type Alias Declarations or References
        elif effective_kind in alias_kinds:
            alias_occurrences[orig_name].append(occurrence_details)
        # Enum Constant Declarations or References
        elif effective_kind in enum_const_kinds:
            enum_const_occurrences[orig_name].append(occurrence_details)
        # Namespace Declarations or References
        elif effective_kind in namespace_kinds:
             namespace_occurrences[orig_name].append(occurrence_details)
        # Label Definitions or References (goto)
        elif effective_kind in label_kinds:
            label_occurrences[orig_name].append(occurrence_details)
        # Type References that didn't resolve to a specific category above
        # Often happens when a typedef/class is used as a type specifier
        elif kind in type_ref_kinds:
            # Try to categorize based on the referenced type if possible
             if referenced:
                ref_kind = referenced.kind
                if ref_kind in class_kinds:
                    class_occurrences[orig_name].append(occurrence_details)
                elif ref_kind in typedef_kinds:
                    typedef_occurrences[orig_name].append(occurrence_details)
                elif ref_kind in alias_kinds:
                    alias_occurrences[orig_name].append(occurrence_details)
                # else: # Fallback if referenced type is unknown/uncategorized
                #    print(f"Debug: Uncategorized TYPE_REF: {orig_name} (Ref kind: {ref_kind.name}) at {occurrence_details['line']}:{occurrence_details['column']}")

        # Other Declaration References (fallback)
        elif kind in decl_ref_kinds:
             # This catches references to entities not covered above
             # Re-check referenced kind for safety
            if referenced:
                ref_kind = referenced.kind
                if ref_kind in var_kinds: # Could be global var ref
                    var_occurrences[orig_name].append(occurrence_details)
                elif ref_kind in fun_kinds:
                     fun_occurrences[orig_name].append(occurrence_details)
                elif ref_kind in enum_const_kinds:
                     enum_const_occurrences[orig_name].append(occurrence_details)
                # else:
                #     print(f"Debug: Uncategorized DECL_REF_EXPR: {orig_name} (Ref kind: {ref_kind.name}) at {occurrence_details['line']}:{occurrence_details['column']}")

        elif effective_kind in template_param_kinds:
            template_param_occurrences[orig_name].append(occurrence_details)
            processed_tokens.add(token_key) # Mark as processed here
            continue # Go to next token
        
        elif effective_kind in function_template_kinds:
            function_template_occurrences[orig_name].append(occurrence_details)
            processed_tokens.add(token_key)
        
        elif effective_kind in class_template_kinds:
            class_template_occurrences[orig_name].append(occurrence_details)
            processed_tokens.add(token_key)

        else:
                # If no referenced cursor, treat as undeclared
            # print(f"Debug: No referenced cursor for {kind}: {orig_name} at {occurrence_details['line']}:{occurrence_details['column']}")
            undeclared_occurrences[orig_name].append(occurrence_details)
        # print(f"Debug: {kind}: {orig_name} at {occurrence_details['line']}:{occurrence_details['column']}")

        # Add token to processed set only if it was categorized
        if orig_name in var_occurrences or orig_name in fun_occurrences or \
           orig_name in method_occurrences or orig_name in field_occurrences or \
           orig_name in class_occurrences or orig_name in typedef_occurrences or \
           orig_name in macro_occurrences or orig_name in enum_const_occurrences or \
           orig_name in alias_occurrences or orig_name in namespace_occurrences or \
           orig_name in label_occurrences or orig_name in param_occurrences:
             processed_tokens.add(token_key)


    # Combine all occurrences into a single dictionary
    all_occurrences = {
        "variables": dict(var_occurrences),
        "parameters": dict(param_occurrences),
        "functions": dict(fun_occurrences),
        "methods": dict(method_occurrences), # Includes constructors/destructors
        "fields": dict(field_occurrences),
        "classes": dict(class_occurrences), # Includes structs, unions, enums
        "typedefs": dict(typedef_occurrences),
        "aliases": dict(alias_occurrences),
        "enum_constants": dict(enum_const_occurrences),
        "macros": dict(macro_occurrences), # Definitions and instantiations
        "namespaces": dict(namespace_occurrences),
        "labels": dict(label_occurrences),
        "template_parameters": dict(template_param_occurrences),
        "function_templates": dict(function_template_occurrences),
        "class_templates": dict(class_template_occurrences),
        "undeclared": dict(undeclared_occurrences),
    }
    
    original_all_occurrences = all_occurrences.copy() # Keep a copy of the original occurrences
    
    for key, value in original_all_occurrences.items():
        # Update the occurrences using the complete_occurrences_with_string_matches function
        value = complete_occurrences_with_string_matches(value, file_path)
        all_occurrences[key] = value
    
    
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
    """
    # Normalize the dictionary as above
    
    """
        "variables": dict(var_occurrences),
        "parameters": dict(param_occurrences),
        "functions": dict(fun_occurrences),
        "methods": dict(method_occurrences), # Includes constructors/destructors
        "fields": dict(field_occurrences),
        "classes": dict(class_occurrences), # Includes structs, unions, enums
        "typedefs": dict(typedef_occurrences),
        "aliases": dict(alias_occurrences),
        "enum_constants": dict(enum_const_occurrences),
        "macros": dict(macro_occurrences), # Definitions and instantiations
        "namespaces": dict(namespace_occurrences),
        "labels": dict(label_occurrences),
        "template_parameters": dict(template_param_occurrences),
        "function_templates": dict(function_template_occurrences),
        "class_templates": dict(class_template_occurrences),
        "undeclared": dict(undeclared_occurrences),
    """
    result_dict = {
        "variables": {}, # merge the variables and parameters
        "func_names": {}, # merge the functions, methods, function_templates
        # keep the rest as is
    }
    for key, value in all_occurrences.items():
        if key in ("variables", "parameters"):
            for name, loc_list in value.items():
                if name not in result_dict["variables"]:
                    result_dict["variables"][name] = []
                result_dict["variables"][name].extend(loc_list)
        elif key in ("functions", "methods", "function_templates"):
            for name, loc_list in value.items():
                if name not in result_dict["func_names"]:
                    result_dict["func_names"][name] = []
                result_dict["func_names"][name].extend(loc_list)
        else:
            result_dict[key] = value
        

    return result_dict # Return occurrences and no error message


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a C/C++ source file and record all identifier occurrences using clang."
                    "Stores the occurrences (name, file, line, column, offsets) in a JSON file."
    )
    parser.add_argument("file", help="Path to the C/C++ source file.")
    parser.add_argument("occurrence_file", help="Path to the output JSON file to store the occurrence data.")
    parser.add_argument("--debug", action="store_true", help="Enable debug output (currently limited effect).") # Keep debug flag if needed later
    args = parser.parse_args()

    print(f"Processing file: {args.file}")
    try:
        occurrences = record_identifier_occurrences(args.file)

        # if error_msg:
            # print(f"Error: {error_msg}")
            # return 1 # Indicate error

        if occurrences is None:
            print(f"Error: Failed to process {args.file} (occurrences is None).")
            return 1

        print(f"Analysis complete. Writing occurrences to {args.occurrence_file}")
        try:
            with open(args.occurrence_file, "w", encoding="utf-8") as f:
                json.dump(occurrences, f, indent=2)
            print(f"Identifier occurrences successfully written to {args.occurrence_file}.")
        except IOError as e:
             print(f"Error: Could not write to output file {args.occurrence_file}: {e}")
             return 1
        except TypeError as e:
             print(f"Error: Could not serialize occurrences to JSON: {e}")
             return 1


        if args.debug:
            print("\nOccurrence Summary:")
            for category, items in occurrences.items():
                if items: # Only print categories with found items
                    count = sum(len(loc_list) for loc_list in items.values())
                    print(f"- {category.capitalize()}: {len(items)} unique names, {count} total occurrences.")

    except Exception as e:
        import traceback
        print(f"An unexpected error occurred during processing:")
        print(traceback.format_exc())
        return 1

    return 0 # Indicate success

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
