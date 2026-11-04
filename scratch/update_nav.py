with open('docs/index.html', 'r', encoding='utf-8') as f:
    c = f.read()
target = '<a href="wsplang.html" style="color: var(--text-muted); text-decoration: none;">WSP-Lang Spec</a>'
repl = '<a href="getting_started.html" style="color: var(--text-muted); text-decoration: none;">Getting Started</a>\n      ' + target
if target in c:
    c = c.replace(target, repl, 1)
    with open('docs/index.html', 'w', encoding='utf-8') as f:
        f.write(c)
    print('index.html updated')
else:
    print('target not found')
