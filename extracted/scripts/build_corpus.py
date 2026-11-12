import ast, os, re, sys, time
roots = ["/usr/lib/python3.12", "/usr/local/lib"]
out = open("corpus.txt", "w")
t0 = time.time(); nfiles = 0; nwords = 0
def emit(txt):
    global nwords
    for para in re.split(r"\n\s*\n", txt):
        w = re.findall(r"[a-z]+", para.lower())
        if len(w) >= 6:
            out.write(" ".join(w) + "\n"); nwords += len(w)
for root in roots:
    for dp, dn, fs in os.walk(root):
        if "test" in dp.lower() or "tests" in dp.lower(): continue
        for f in fs:
            if not f.endswith(".py"): continue
            try:
                src = open(os.path.join(dp, f), encoding="utf8", errors="ignore").read()
                tree = ast.parse(src)
            except Exception: continue
            nfiles += 1
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    d = ast.get_docstring(node)
                    if d: emit(d)
    if time.time() - t0 > 240: break
# pydoc topics + system docs
try:
    from pydoc_data import topics
    for v in topics.topics.values(): emit(v)
except Exception as e: print("topics", e)
for dp, dn, fs in os.walk("/usr/share/doc"):
    for f in fs:
        if f.endswith((".txt", ".md", ".rst")) or f in ("README", "NEWS", "changelog"):
            try: emit(open(os.path.join(dp, f), encoding="utf8", errors="ignore").read())
            except Exception: pass
out.close()
print("files", nfiles, "words", nwords, "secs", round(time.time() - t0))
