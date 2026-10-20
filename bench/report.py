#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only WITH AdditionRef-Linking-Attribution-Exception
# Copyright (C) 2026 srose69. E5M2EffMA; see NOTICE, LICENSE and LICENSE-EXCEPTION (additional terms).
"""Turn a bench/run_bench.sh result directory into benches/<gpu>.md and its row in benches/table.md.

usage: bench/report.py <result dir> [<benches dir>]
The result directory holds env.txt (key=value lines), bench_gemm.txt, bench_gemv.txt,
bench_adamw.txt and clocks_<bench>.csv as written by bench/run_bench.sh.
"""
import csv, os, re, sys

def slug(name):
    words = [w for w in re.split(r'\s+', name.strip()) if w.lower() not in ('nvidia', 'geforce', 'tesla', 'quadro', 'gtx', 'rtx')]
    return re.sub(r'[^a-z0-9]+', '-', ' '.join(words).lower()).strip('-') or 'gpu'

def read_env(path):
    env = {}
    for line in open(path):
        if '=' in line:
            k, v = line.split('=', 1); env[k.strip()] = v.strip()
    return env

ROW = re.compile(r'^(?P<name>\S.*?)\s{2,}(?P<ms>[\d.]+) ms\s+(?P<rate>[\d.]+) (?P<unit>GMAC/s|GB/s)(?P<rest>.*)$')

def gemm_rows(path):
    rows = {}
    for line in open(path):
        m = ROW.match(line.rstrip())
        if not m: continue
        rest = m.group('rest')
        mhz = re.search(r'([\d.]+) MHz', rest); pct = re.search(r'([\d.]+)%', rest)
        rows[m.group('name').strip()] = dict(ms=float(m.group('ms')), gmacs=float(m.group('rate')),
                                             mhz=float(mhz.group(1)) if mhz else None, pct=float(pct.group(1)) if pct else None)
    return rows

def gemv_rows(path):
    rows, head = {}, {}
    for line in open(path):
        h = re.search(r'DRAM peak ([\d.]+) GB/s(?: \((\d+)-bit bus\))?, D2D copy ([\d.]+) GB/s', line)
        if h: head = dict(dram=float(h.group(1)), bus=h.group(2), copy=float(h.group(3)))
        m = ROW.match(line.rstrip())
        if not m: continue
        p = re.findall(r'([\d.]+)%', m.group('rest')); mhz = re.search(r'([\d.]+) MHz', m.group('rest'))
        rows[m.group('name').strip()] = dict(ms=float(m.group('ms')), gbs=float(m.group('rate')), dram=float(p[0]),
                                             copy=float(p[1]), mhz=float(mhz.group(1)) if mhz else None)
    return rows, head

def adam_rows(path):
    rows = {}
    for line in open(path):
        m = re.search(r'state (\w+)( \+ E5M2 copy)?:\s+([\d.]+) ms\s+([\d.]+) GB/s\s+([\d.]+)% of DRAM peak\s+([\d.]+) MHz', line)
        if m: rows[m.group(1) + (' + E5M2 copy' if m.group(2) else '')] = dict(ms=float(m.group(3)), gbs=float(m.group(4)), dram=float(m.group(5)))
    return rows

def torch_adam_rows(path):
    """bench_torch_adamw.py output: {label: (ms, speed-up vs torch fused)}"""
    rows = {}
    if not os.path.exists(path): return rows
    for line in open(path):
        m = re.match(r'^(?P<name>\S.*?)\s{2,}(?P<ms>[\d.]+) ms\s+(?P<x>[\d.]+)×', line.rstrip())
        if m: rows[m.group('name').strip()] = (float(m.group('ms')), float(m.group('x')))
    return rows

def clocks(path):
    """SM / memory clock, temperature, power and throttle reasons over the busy samples of a benchmark's log"""
    sm, mem, temp, watt, reasons = [], [], [], [], {}
    if not os.path.exists(path): return None
    for r in csv.reader(open(path), skipinitialspace=True):
        if len(r) < 6 or int(r[5], 16) & 0x1: continue  # idle samples (the cool-down pauses)
        try:
            sm.append(float(r[1].split()[0])); mem.append(float(r[2].split()[0]))
            temp.append(float(r[3])); watt.append(float(r[4].split()[0]))
        except ValueError: continue
        reasons[r[5]] = reasons.get(r[5], 0) + 1
    if not sm: return None
    capped = sum(n for k, n in reasons.items() if int(k, 16) & 0x4)
    thermal = sum(n for k, n in reasons.items() if int(k, 16) & 0x60)
    s = sorted(sm)  # 5th percentile: a few samples catch the clock dropping into a cool-down pause
    return dict(sm=(s[len(s) // 20], s[len(s) // 2], s[-1]), mem=sorted(set(mem)), temp=(min(temp), max(temp)),
                watt=max(watt), n=len(sm), capped=capped, thermal=thermal)

def fmt_clock(c):
    if not c: return '—'
    return (f"SM median {c['sm'][1]:.0f} MHz ({c['sm'][0]:.0f}–{c['sm'][2]:.0f}), memory {'/'.join(f'{x:.0f}' for x in c['mem'])} MHz, "
            f"{c['temp'][0]:.0f}–{c['temp'][1]:.0f} °C, up to {c['watt']:.0f} W; power-capped in {c['capped']} of {c['n']} samples, "
            f"thermal slowdown in {c['thermal']}")

def main(res, out_dir):
    env = read_env(os.path.join(res, 'env.txt'))
    gpu = env.get('name', 'unknown GPU'); key = env.get('file') or slug(gpu)
    g = gemm_rows(os.path.join(res, 'bench_gemm.txt'))
    v, vh = gemv_rows(os.path.join(res, 'bench_gemv.txt'))
    a = adam_rows(os.path.join(res, 'bench_adamw.txt'))
    cl = {b: clocks(os.path.join(res, f'clocks_bench_{b}.csv')) for b in ('gemm', 'gemv', 'adamw')}
    cub = g.get('cuBLAS SGEMM FP32 -> FP32, TN') or g.get('cuBLAS SGEMM TN FP32 -> FP32')
    # SM count and bus width come from cudaDeviceProp via the benchmark headers (nvidia-smi -q no
    # longer reports them on recent drivers)
    sms = re.search(r', (\d+) SMs;', open(os.path.join(res, 'bench_gemm.txt')).read())
    if sms and not env.get('sms'): env['sms'] = sms.group(1)
    if vh.get('bus') and 'bus' not in env.get('memory', ''): env['memory'] = f"{env.get('memory', '?')}, {vh['bus']}-bit bus"
    L = [f"# {gpu}", '',
         f"Results of `bench/run_bench.sh`. Methodology and how to read them: [docs/performance.md](../docs/performance.md).", '',
         '| | |', '|---|---|']
    for k, label in [('name', 'GPU'), ('compute_cap', 'Compute capability'), ('sms', 'SMs'), ('memory', 'Memory'),
                     ('driver', 'Driver'), ('cuda', 'CUDA toolkit'), ('library', 'E5M2EffMA'), ('date', 'Date'),
                     ('power_limit', 'Power limit'), ('clock_lock', 'Clock settings'), ('cooling', 'Cooling'), ('notes', 'Notes')]:
        if env.get(k): L.append(f'| {label} | {env[k]} |')
    for b in ('gemm', 'gemv', 'adamw'):
        if cl[b]: L.append(f'| During bench_{b} | {fmt_clock(cl[b])} |')
    L += ['', '## GEMM, M = N = K = 4096', '',
          '| Operation | Time | GMAC/s | SM clock | % of FP32 peak | vs cuBLAS SGEMM TN |', '|---|---:|---:|---:|---:|---:|']
    for name, r in g.items():
        ratio = f"{cub['ms'] / r['ms']:.2f}×" if cub and not name.startswith('cuBLAS') else '—'
        L.append(f"| {name} | {r['ms']:.2f} ms | {r['gmacs']:.0f} | {r['mhz']:.0f} MHz | {r['pct']:.1f} % | {ratio} |" if r['mhz']
                 else f"| {name} | {r['ms']:.2f} ms | {r['gmacs']:.0f} | — | — | {ratio} |")
    L += ['', f"## GEMV, 8192 × 8192 E5M2 weights (DRAM peak {vh.get('dram', 0):.0f} GB/s, device-to-device copy {vh.get('copy', 0):.0f} GB/s)", '',
          '| Operation | Time | GB/s over the weights | % of DRAM peak | % of copy | SM clock |', '|---|---:|---:|---:|---:|---:|']
    for name, r in v.items():
        L.append(f"| {name} | {r['ms']:.3f} ms | {r['gbs']:.0f} | {r['dram']:.0f} % | {r['copy']:.0f} % | {r['mhz']:.0f} MHz |")
    L += ['', '## Adam / AdamW, 64 M parameters', '', '| State | Time | GB/s | % of DRAM peak |', '|---|---:|---:|---:|']
    for name, r in a.items():
        L.append(f"| {name} | {r['ms']:.2f} ms | {r['gbs']:.0f} | {r['dram']:.0f} % |")
    ta = torch_adam_rows(os.path.join(res, 'bench_torch_adamw.txt'))
    if ta:
        head = [l.strip() for l in open(os.path.join(res, 'bench_torch_adamw.txt')) if 'torch' in l and 'e5m2effma' in l][:1]
        L += ['', '## AdamW vs torch.optim.AdamW, 64 M parameters', '']
        if head: L += [f'`bench/bench_torch_adamw.py` — {head[0]}', '']
        L += ['| Optimizer | Params | Layout | Step time | vs torch fused |', '|---|---|---|---:|---:|']
        for name, (ms, x) in ta.items():
            opt, dt, layout = [t.strip() for t in name.rsplit(',', 2)]
            L.append(f"| {opt} | {dt.replace(' params', '')} | {layout} | {ms:.2f} ms | {x:.2f}× |")
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, f'{key}.md'), 'w').write('\n'.join(L) + '\n')

    # summary row in benches/table.md
    tn = g.get('e5ma_gemm TN E5M2 -> BF16'); mix = g.get('e5ma_gemm TN E5M2 x BF16 -> BF16')
    v1 = v.get('e5ma gemv n=1 x=E5M2 -> BF16') or v.get('e5ma gemv n=1 x=BF16 -> BF16')
    sv = v.get('cuBLAS SGEMV FP32 (same M, K)'); af = a.get('FP32')
    def pct(r): return f"{r['pct']:.0f} %" if r and r.get('pct') is not None else '—'
    def speedup(base, r, digits=2): return f"{base['ms'] / r['ms']:.{digits}f}×" if base and r else '—'
    def bw(r): return f"{r['gbs']:.0f} GB/s ({r['dram']:.0f} %)" if r else '—'
    def vs_torch(moments, params):
        r = ta.get(f'e5m2effma AdamW, {moments} moments, {params} params, 1 x 64M')
        return f'{r[1]:.2f}×' if r else '—'
    row = (f"| [{gpu}]({key}.md) | {env.get('compute_cap', '—')} | {pct(tn)} vs {pct(cub)} | {speedup(cub, tn)} "
           f"| {speedup(cub, mix)} | {bw(v1)} | {speedup(sv, v1, 1)} | {bw(af)} "
           f"| {vs_torch('FP32', 'FP32')} / {vs_torch('BF16', 'BF16')} | {env.get('date', '—')} |")
    tpath = os.path.join(out_dir, 'table.md')
    head = ['# Results by GPU', '',
            'One row per GPU, produced by `bench/run_bench.sh` (see [docs/performance.md](../docs/performance.md)); the',
            'linked file has the full tables and the conditions of the run. Percentages of the FP32 peak use the SM',
            'clock measured during each row; the baseline is cuBLAS on FP32 data of the same shape.', '',
            '| GPU | CC | GEMM TN E5M2: % of FP32 peak (vs cuBLAS SGEMM) | GEMM TN E5M2 × E5M2 vs SGEMM | GEMM TN E5M2 × BF16 vs SGEMM '
            '| GEMV n = 1 (% of DRAM peak) | GEMV n = 1 vs SGEMV | AdamW FP32 state (% of DRAM peak) '
            '| AdamW vs torch fused (FP32 / BF16 params and state) | Date |',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---|']
    rows = []
    if os.path.exists(tpath):
        rows = [l for l in open(tpath).read().split('\n') if l.startswith('| [') and f']({key}.md)' not in l]
    rows.append(row)
    open(tpath, 'w').write('\n'.join(head + sorted(rows)) + '\n')
    print(f'wrote {os.path.join(out_dir, key + ".md")} and its row in {tpath}')

if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benches'))
