import sys
from tree_sitter import Language, Parser
import tree_sitter_java 

def parse_java(java_code_text):
    output = {
        "variables": {},
        "func_names": {}
    }
    
    java_code = java_code_text.encode('utf8')

    # load & set up parser correctly
    JAVA_LANG = Language(tree_sitter_java.language())
    parser    = Parser(JAVA_LANG)

    tree = parser.parse(java_code)
    root = tree.root_node

    imports, types_, methods, ctors, ctor_calls, vars_, params, usages, type_uses = (
        set(), [], [], [], [], [], [], [], []
    )

    def walk(node):
        t = node.type

        # skip package names
        if t == "package_declaration":
            return

        # imports
        if t == "import_declaration":
            path_node = next(
                (c for c in node.children if c.type == "scoped_identifier"),
                None
            )
            if path_node:
                pkg = get_text(path_node).split('.')[-1]
                imports.add(pkg)
            return

        # type declarations
        if t == "class_declaration":
            types_.append((node.child_by_field_name("name"), "class_decl"))
        elif t == "interface_declaration":
            types_.append((node.child_by_field_name("name"), "interface_decl"))
        elif t == "enum_declaration":
            types_.append((node.child_by_field_name("name"), "enum_decl"))

        # class‐level fields
        if t == "field_declaration":
            for decl in node.children:
                if decl.type == "variable_declarator":
                    vars_.append(decl.child_by_field_name("name"))
            return

        # constructors
        if t == "constructor_declaration":
            ctors.append(node.child_by_field_name("name"))
            # still recurse into parameters & body
            
        # ─── constructor calls (new ClassName(...)) ─────────────
        if t == "object_creation_expression":
            # in tree-sitter-java this has a child field “type”
            type_node = next(
                (c for c in node.children
                 if c.type in ("type_identifier", "scoped_type_identifier")),
                None
            )            
            if type_node:
                ctor_calls.append(type_node)
            # if you only care about the ctor call, you can return here
            # otherwise you can recurse into the arguments:
            for c in node.children:
                walk(c)
            return
        # methods
        if t == "method_declaration":
            methods.append(node.child_by_field_name("name"))
            # still recurse into parameters & body


        if t == "enhanced_for_statement":
            nm = node.child_by_field_name("name")
            if nm:
                vars_.append(nm)
                type_node = nm.child_by_field_name("type")
                if type_node:
                    if type_node.type in ("type_identifier","scoped_type_identifier"):
                        # e.g. `Test`
                        type_uses.append(type_node)
        # parameters
        elif t == "formal_parameter":
            params.append(node.child_by_field_name("name"))
            return

        # enum constants
        elif t == "enum_constant":
            vars_.append(node.child_by_field_name("name"))
            return

        # local vars
        elif t == "local_variable_declaration":
    # 1) pull out the declared type (handles both leaf & generic cases)
            type_node = node.child_by_field_name("type")
            if type_node:
                if type_node.type in ("type_identifier","scoped_type_identifier"):
                    # e.g. `Test`
                    type_uses.append(type_node)
                else:
                    # e.g. `List<String>` → find the inner identifier
                    id_node = next(
                        (c for c in type_node.children
                        if c.type in ("type_identifier","scoped_type_identifier")),
                        None
                    )
                    if id_node:
                        type_uses.append(id_node)

            # 2) now record the actual variable‐names
            for decl in node.children:
                if decl.type == "variable_declarator":
                    vars_.append(decl.child_by_field_name("name"))
                    # return

        # all other identifiers → usages
        elif t == "identifier":
            name = get_text(node)
            if name not in imports:
                usages.append(node)

        # recurse
        for c in node.children:
            walk(c)

    def get_text(node):
        return java_code[node.start_byte:node.end_byte].decode('utf8')
    
    walk(root)

    # build a map: name → list of (kind, line, col)
    occurrences = {}

    def record(nodes, kind):
        for n in nodes:
            name = get_text(n)
            occurrences.setdefault(name, []).append((
                kind,
                n.start_point[0] + 1,
                n.start_point[1] + 1,
                n.start_byte,
                n.end_byte,
            ))

    # record all the things
    for node, kind in types_:
        record([node], kind)
    record(ctors,    "constructor_decl")
    record(methods,  "method_decl")
    record(vars_,    "var_decl")
    record(params,   "param_decl")
    declared = { get_text(n) for n, kind in types_ }
    type_uses = [n for n in type_uses
                if get_text(n) in declared]
    ctor_calls = [n for n in ctor_calls
                if get_text(n) in declared]
    record(ctor_calls,  "ctor_call")
    record(type_uses, "type_use")
    
    # record usages without duplicating locations
    for n in usages:
        name = get_text(n)
        if name not in occurrences:
            continue
        line, col = n.start_point[0] + 1, n.start_point[1] + 1
        start_offset = n.start_byte
        end_offset = n.end_byte
        seen = {(l, c) for _, l, c, _, _ in occurrences[name]}
        if (line, col) not in seen:
            occurrences[name].append(("usage", line, col, start_offset, end_offset))

    def deduplicate_occurences(occs):
        """Remove duplicate occurrences from the list."""
        seen = set()
        deduped = []
        for kind, line, col, start_offset, end_offset in occs:
            if (line, col) not in seen:
                seen.add((line, col))
                deduped.append((kind, line, col, start_offset, end_offset))
        return deduped

    # print summary
    # for name, occs in sorted(occurrences.items()):
    #     print(f"\n{name!r}:")
    #     for kind, line, col, _, _ in occs:
    #         print(f"  • {kind:12s} at line {line}, col {col}")
            
    for name, occs in sorted(occurrences.items()):
        # Turn each (kind, line, col) into your dict
        entries = [
            {"loc": [start_offset, end_offset], "text": name}
            for kind, line, col, start_offset, end_offset in deduplicate_occurences(occs)
        ]

        # If at least one occurrence is a function declaration, treat it as func_names
        if any(kind == "function_decl" for kind, _, _, _, _ in deduplicate_occurences(occs)):
            output["func_names"].setdefault(name, []).extend(entries)
        else:
            # Otherwise treat it as a variable
            output["variables"].setdefault(name, []).extend(entries)
    return output

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} path/to/file.java")
        sys.exit(1)
    fname = sys.argv[1]
    if not fname.endswith('.java'):
        print("Warning: file does not end in .java")
    parse_java(fname)
