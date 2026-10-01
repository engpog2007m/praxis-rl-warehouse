#!/usr/bin/env python3
"""Post-copy processing for collect_source.sh.

Operates only on files already copied into the repository. Performs:

  1. Path portability  - rewrites hard-coded ~/Praxis paths to a PRAXIS_ROOT
                         lookup. When PRAXIS_ROOT is unset the resolved path
                         is byte-for-byte identical to the original, so the
                         substitution cannot change behaviour on the author's
                         machine.
  2. Model-name fix    - train_gazebo_ppo.py was left pointing at an unused
                         v5/v6 filename during H3 experiments; evaluation
                         scripts load ppo_gazebo_v4. Normalised to v4.
  3. Verification      - py_compile every module, detect tab indentation,
                         report residual non-portable paths.
  4. Parameter audit   - prints the environment constants that determine
                         the reported results so a human can confirm them.
  5. Provenance        - writes docs/PROVENANCE.md mapping each script to
                         the artifacts it reads and writes (heuristic).

Exit status is non-zero if any module fails to compile.
"""
import argparse
import pathlib
import py_compile
import re
import sys

ROOT_EXPR = "os.environ.get('PRAXIS_ROOT', os.path.expanduser('~/Praxis'))"
EXPAND_PAT = re.compile(
    r"""os\.path\.expanduser\(\s*(?P<f>[fF]?)(?P<q>['"])~/Praxis"""
    r"""(?P<rest>(?:/[^'"]*)?)(?P=q)\s*\)"""
)
ARTIFACT_PAT = re.compile(r"""['"]([^'"\s]*?\.(?:csv|zip|pkl|png|json|txt|sdf))['"]""")
WRITE_HINTS = ('to_csv(', '.save(', 'savefig(', 'json.dump(', 'pickle.dump(',
               "'w')", '"w")', "'wb')", '"wb")')
READ_HINTS = ('read_csv(', '.load(', 'json.load(', 'pickle.load(', "'rb')",
              '"rb")', 'imread(', 'Image.open(')

CANONICAL_SIM = {'data_registry.py', 'warehouse_env.py', 'gazebo_env.py',
                 'gazebo_env_lidar.py', '__init__.py'}

EXPECTED = {
    'gazebo_env.py': {
        'goals': {1: 3.0, 2: 3.5, 3: 2.5, 4: 1.2, 1001: 1.5},
        'max_steps': {200},
        'fragile': 0.5, 'heavy': 0.7,
    },
    'gazebo_env_lidar.py': {
        'goals': {1: 2.0, 2: 2.0, 3: 2.0, 4: 2.0, 1001: 2.0},
        'max_steps': {200, 500},   # scripts pass max_steps=200 explicitly
        'fragile': 0.5, 'heavy': 0.7,
        'arrival': 0.5, 'collision': 0.30, 'lateral': 0.75,
    },
}


def rewrite_paths(text):
    out, total = [], 0
    for line in text.splitlines(keepends=True):
        if 'PRAXIS_ROOT' in line:
            out.append(line)
            continue

        def repl(m):
            rest = m.group('rest').lstrip('/')
            if not rest:
                return ROOT_EXPR
            q = m.group('q')
            return f"os.path.join({ROOT_EXPR}, {m.group('f')}{q}{rest}{q})"

        new, n = EXPAND_PAT.subn(repl, line)
        total += n
        out.append(new)
    return ''.join(out), total


def normalise_model_version(path):
    if path.name != 'train_gazebo_ppo.py':
        return None
    text = path.read_text(encoding='utf-8')
    new = re.sub(r'ppo_gazebo_v[56]\b', 'ppo_gazebo_v4', text)
    new = re.sub(r'gazebo_training_curves_v[56]\.csv',
                 'gazebo_training_curves_v4.csv', new)
    if new != text:
        path.write_text(new, encoding='utf-8')
        return 'normalised model/curve filenames to v4'
    return 'already v4 (no change)'


def tab_lines(text):
    return [i + 1 for i, ln in enumerate(text.splitlines())
            if ln[:len(ln) - len(ln.lstrip())].count('\t')]


def provenance(path, text):
    reads, writes = set(), set()
    for ln in text.splitlines():
        names = {pathlib.PurePosixPath(n).name for n in ARTIFACT_PAT.findall(ln)}
        names.discard('')
        if not names:
            continue
        if any(h in ln for h in WRITE_HINTS):
            writes |= names
        elif any(h in ln for h in READ_HINTS):
            reads |= names
    return sorted(reads), sorted(writes)


def audit_params(path):
    text = path.read_text(encoding='utf-8')
    found = {}
    block = re.search(r'GOAL_OFFSETS\s*=\s*\{(.*?)\n\s*\}', text, re.S)
    if block:
        found['goals'] = {int(k): float(v) for k, v in re.findall(
            r'(\d+)\s*:\s*np\.array\(\[\s*([\d.]+)', block.group(1))}
    m = re.search(r'max_steps\s*=\s*(\d+)', text)
    if m:
        found['max_steps'] = int(m.group(1))
    m = re.search(r'FRAGILE_THRESHOLD\s*=\s*([\d.]+)', text) or \
        re.search(r"is_fragile'\]\s*and\s*speed\s*>\s*([\d.]+)", text)
    if m:
        found['fragile'] = float(m.group(1))
    m = re.search(r'HEAVY_THRESHOLD\s*=\s*([\d.]+)', text) or \
        re.search(r"is_heavy'\]\s*and\s*speed\s*>\s*([\d.]+)", text)
    if m:
        found['heavy'] = float(m.group(1))
    m = re.search(r'ARRIVAL_RADIUS\s*=\s*([\d.]+)', text)
    if m:
        found['arrival'] = float(m.group(1))
    m = re.search(r'collision_dist\s*=\s*([\d.]+)', text)
    if m:
        found['collision'] = float(m.group(1))
    m = re.search(r'uniform\(\s*-([\d.]+)\s*,\s*([\d.]+)\s*\)', text)
    if m:
        found['lateral'] = float(m.group(2))
    return found


def compare(name, found, exp):
    lines, ok = [], True
    for key, want in exp.items():
        got = found.get(key)
        if key == 'max_steps':
            good = got in want
        elif isinstance(want, dict):
            good = got == want
        else:
            good = got is not None and abs(got - want) < 1e-9
        ok &= good
        lines.append(f"    {'OK  ' if good else 'WARN'} {key:<10} found={got}  expected={want}")
    return ok, lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repo', required=True)
    args = ap.parse_args()
    repo = pathlib.Path(args.repo).resolve()

    py_files = sorted(
        [p for p in (repo / 'src').rglob('*.py')] +
        [p for p in (repo / 'scripts').glob('*.py')] +
        [p for p in repo.glob('*.py')]
    )

    compile_fail, tab_warn, residual_expand, residual_home = [], [], [], []
    rewrites = 0
    prov_rows = []

    print('\n[1/5] Path portability')
    for p in py_files:
        text = p.read_text(encoding='utf-8')
        new, n = rewrite_paths(text)
        if n:
            p.write_text(new, encoding='utf-8')
            rewrites += n
            print(f'    rewrote {n:>2} path(s)  {p.relative_to(repo)}')
    print(f'    total rewrites: {rewrites}')

    print('\n[2/5] Model-name normalisation')
    for p in py_files:
        msg = normalise_model_version(p)
        if msg:
            print(f'    {p.relative_to(repo)}: {msg}')

    print('\n[3/5] Verification')
    for p in py_files:
        text = p.read_text(encoding='utf-8')
        try:
            py_compile.compile(str(p), doraise=True)
        except py_compile.PyCompileError as e:
            compile_fail.append((p, str(e).splitlines()[-1]))
        tl = tab_lines(text)
        if tl:
            tab_warn.append((p, tl[:5]))
        if re.search(r"expanduser\([^)]*~/Praxis", text.replace(ROOT_EXPR, '')):
            residual_expand.append(p)
        if '/home/' in text:
            residual_home.append(p)
        reads, writes = provenance(p, text)
        prov_rows.append((p.relative_to(repo), reads, writes))

    print(f'    modules compiled: {len(py_files) - len(compile_fail)}/{len(py_files)}')
    for p, err in compile_fail:
        print(f'    FAIL compile  {p.relative_to(repo)}: {err}')
    for p, tl in tab_warn:
        print(f'    WARN tabs     {p.relative_to(repo)} (lines {tl})')
    for p in residual_expand:
        print(f'    WARN path     {p.relative_to(repo)} still has a non-portable '
              f'~/Praxis expanduser (manual edit needed)')
    for p in residual_home:
        print(f'    WARN path     {p.relative_to(repo)} contains an absolute /home/ path')

    for sdf in sorted((repo / 'worlds').glob('*.sdf')):
        if '/home/' in sdf.read_text(encoding='utf-8', errors='replace'):
            print(f'    WARN path     {sdf.relative_to(repo)} contains an absolute '
                  f'/home/ path; use model:// URIs + GZ_SIM_RESOURCE_PATH')

    print('\n[4/5] Environment parameter audit')
    for name, exp in EXPECTED.items():
        path = repo / 'src' / 'simulation' / name
        if not path.exists():
            print(f'  {name}: MISSING')
            continue
        ok, lines = compare(name, audit_params(path), exp)
        print(f"  {name}: {'matches reported configuration' if ok else 'REVIEW BELOW'}")
        for ln in lines:
            print(ln)

    print('\n[5/5] Provenance')
    docs = repo / 'docs'
    docs.mkdir(exist_ok=True)
    with open(docs / 'PROVENANCE.md', 'w', encoding='utf-8') as f:
        f.write('# Artifact Provenance\n\n')
        f.write('Generated by `tools/collect_helper.py`. Heuristic: a script is '
                'listed as reading or writing an artifact when a quoted filename '
                'appears on the same line as a read or write call. Verify any '
                'entry you rely on by reading the script.\n\n')
        f.write('| Script | Reads | Writes |\n|---|---|---|\n')
        for rel, reads, writes in prov_rows:
            if reads or writes:
                f.write(f"| `{rel}` | {', '.join(reads) or '-'} | "
                        f"{', '.join(writes) or '-'} |\n")
    print(f'    wrote {docs / "PROVENANCE.md"}')

    return 1 if compile_fail else 0


if __name__ == '__main__':
    sys.exit(main())
