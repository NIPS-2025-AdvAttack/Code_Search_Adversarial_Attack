import sys
from tree_sitter import Language, Parser
import tree_sitter_javascript
# --- a set of ECMA-262 globals you don’t want to record ---
BUILTINS = {
    "Math", "Array", "Object", "String", "Number", "Boolean",
    "RegExp", "Date", "Promise", "Map", "Set", "WeakMap", "WeakSet",
    "Symbol", "JSON", "console", "window", "document",
    "parseInt", "parseFloat", "isNaN", "isFinite", "eval",
    "decodeURI", "encodeURI", "decodeURIComponent", "encodeURIComponent",
    "escape", "unescape", "Error", "TypeError", "Intl", "Reflect", "Proxy"
}

def parse_js(js_code):
    output = {
        "variables": {},
        "func_names": {}
    }

    # with open(fname, 'rb') as f:
    #     js_code = f.read()

    js_code = js_code.encode('utf8')

    # --- load & set up parser ---
    JS_LANG = Language(tree_sitter_javascript.language())
    parser = Parser(JS_LANG)

    tree = parser.parse(js_code)
    root = tree.root_node

    imports    = set()
    classes    = []
    functions  = []
    methods    = []
    ctors      = []
    ctor_calls = []
    vars_      = []
    params     = []
    usages     = []

    def get_text(node):
        return js_code[node.start_byte:node.end_byte].decode('utf8')

    def walk(node):
        t = node.type

        # --- imports ---
        if t == "import_declaration":
            # e.g. import Foo, { Bar as B } from "mod";
            clause = node.child_by_field_name("import_clause")
            if clause:
                # default import
                default = clause.child_by_field_name("name")
                if default:
                    imports.add(get_text(default))
                # named imports
                named = clause.child_by_field_name("named_imports")
                if named:
                    for spec in named.named_children:
                        if spec.type == "import_specifier":
                            alias = spec.child_by_field_name("alias") \
                                 or spec.child_by_field_name("name")
                            if alias:
                                imports.add(get_text(alias))
                        elif spec.type == "identifier":
                            imports.add(get_text(spec))
            return

        # --- CommonJS requires ---
        if t == "variable_declaration":
            # look for: const X = require("mod");
            for decl in node.named_children:
                if decl.type == "variable_declarator":
                    name_node = decl.child_by_field_name("name")
                    init = decl.child_by_field_name("value")
                    if init and init.type == "call_expression":
                        callee = init.child_by_field_name("function")
                        arg0   = init.named_children[1] if len(init.named_children) > 1 else None
                        if callee and get_text(callee) == "require" and arg0 and arg0.type == "string":
                            # treat name_node as import
                            if name_node:
                                imports.add(get_text(name_node))
            # continue on to record variables too
        if t in ("public_field_definition", "private_field_definition", "property_definition"):
            # field name lives in the "name" field (or sometimes "key")
            name_node = node.child_by_field_name("name") or node.child_by_field_name("key")
            if name_node:
                vars_.append(name_node)
            return

        if t in ("for_of_statement", "for_in_statement"):
            # child called "left" is either an identifier or a variable_declaration
            left = node.child_by_field_name("left")
            if left and left.type == "identifier":
                usages.append(left)

        if t == "member_expression":
            # child_by_field_name("object") → the thing before the dot
            # child_by_field_name("property") → the identifier after the dot
            obj  = node.child_by_field_name("object")
            if obj and obj.type == "identifier" and get_text(obj) in BUILTINS:
                # e.g. Math.abs → skip both 'Math' and 'abs'
                return
            prop = node.child_by_field_name("property")
            if obj and prop and obj.type == "this" and prop.type == "identifier":
                # Is this on the left‐hand side of an assignment? 
                parent = node.parent
                if parent and parent.type == "assignment_expression" \
                    and parent.child_by_field_name("left") is node:
                    # this.foo = … → treat `foo` as a field declaration
                    vars_.append(prop)
                else:
                    # any other this.foo → usage of that field
                    usages.append(prop)
                # we handled this node fully
                return

        # --- class declaration ---
        if t == "class_declaration":
            name = node.child_by_field_name("name")
            if name:
                classes.append(name)
            # still recurse into body
        # --- function declaration ---
        if t == "function_declaration":
            name = node.child_by_field_name("name")
            if name:
                functions.append(name)
            # recurse into params & body

        # --- method definitions & constructors in classes ---
        if t == "method_definition":
            key = node.child_by_field_name("name")
            if get_text(key) == "constructor":
                # return
                pass
            elif key:
                methods.append(key)
                # detect constructor
                # if get_text(key) == "constructor":
                #     ctors.append(key)

        # --- new expressions (ctor calls) ---
        if t == "new_expression":
            # child 'constructor' may be identifier or member_expression
            cons = node.child_by_field_name("constructor")
            name = get_text(cons)
            if name in BUILTINS:
                pass
                # e.g. new Math() → skip both 'Math' and the constructor
                # return
            if cons and cons.type == "identifier":
                ctor_calls.append(cons)
            # recurse only into args
            for c in node.children:
                walk(c)
            return

        # --- variable declarators ---
        if t == "variable_declarator":
            name = node.child_by_field_name("name")
            if name:
                vars_.append(name)

        # --- function parameters ---
        if t == "formal_parameters":
            for c in node.children:
                if c.type == "identifier":
                    params.append(c)
            # return

        # --- identifiers → usages (fallback) ---
        if t == "identifier":
            name = get_text(node)
            if name in imports or name in BUILTINS:
                return

            if name not in imports:
                usages.append(node)

        # recurse
        for c in node.children:
            walk(c)

    walk(root)

    # --- build occurrences map ---
    occurrences = {}
    def record(nodes, kind):
        for n in nodes:
            nm = get_text(n)
            occurrences.setdefault(nm, []).append((
                kind,
                n.start_point[0] + 1,
                n.start_point[1] + 1,
                n.start_byte,
                n.end_byte  
            ))

    # record all declarations & calls
    record(classes,   "class_decl")
    record(functions, "function_decl")
    record(ctors,     "constructor_decl")
    record(methods,   "method_decl")
    record(vars_,     "var_decl")
    record(params,    "param_decl")
    record(usages,    "usage")

    # only keep ctor_calls for user-declared classes
    declared_classes = { get_text(n) for n in classes }
    filtered_ctor_calls = [n for n in ctor_calls
                           if get_text(n) in declared_classes]
    record(filtered_ctor_calls, "ctor_call")

    # record usages (only for names we've seen declared)
    for n in usages:
        nm = get_text(n)
        if nm in occurrences:
            line, col = n.start_point[0] + 1, n.start_point[1] + 1
            start_offset = n.start_byte
            end_offset = n.end_byte
            seen = {(l, c) for _, l, c, _, _ in occurrences[nm]}
            if (line, col) not in seen:
                occurrences[nm].append(("usage", line, col, start_offset, end_offset))



    def deduplicate_occurences(occs):
        """Remove duplicate occurrences from the list."""
        seen = set()
        deduped = []
        for kind, line, col, start_offset, end_offset in occs:
            if (line, col) not in seen:
                seen.add((line, col))
                deduped.append((kind, line, col, start_offset, end_offset))
        return deduped

    # --- print summary ---
    # for name, occs in sorted(occurrences.items()):
    #     print(f"\n{name!r}:")
    #     for kind, line, col in deduplicate_occurences(occs):
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
        print(f"Usage: {sys.argv[0]} path/to/file.js")
        sys.exit(1)
    fname = sys.argv[1]
    if not fname.endswith('.js'):
        print("Warning: file does not end in .js")
    parse_js(fname)
