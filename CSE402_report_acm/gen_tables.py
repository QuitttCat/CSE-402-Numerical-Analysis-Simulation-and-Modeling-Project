#!/usr/bin/env python3
"""Generate the table files for the CSE 402 ACM-format report directly from the
repo's saved scan / interpolation JSON, so no number is ever transcribed by hand.

Every table is written to its own file, tables/<label>.tex (':' becomes '_'),
so report.tex can \\input each one next to the paragraph that discusses it.
Narrow tables are sized to one column, wide ones span both columns."""
import json, glob, os, math, csv, re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
OUT = os.path.join(HERE, "tables", "_all.tex")

# tables that need the full page width (everything else fits one column)
WIDE = {"tab:campaign", "tab:vqeacc", "tab:interp", "tab:geom", "tab:data:h2:bond"}

MOL_ORDER = ["H2", "LiH", "BeH2", "H2O", "NH3", "N2", "O2", "F2", "H2O2", "NO2", "N2O"]
TEX_MOL = {"H2": r"\ce{H2}", "LiH": r"\ce{LiH}", "BeH2": r"\ce{BeH2}", "H2O": r"\ce{H2O}",
           "NH3": r"\ce{NH3}", "N2": r"\ce{N2}", "O2": r"\ce{O2}", "F2": r"\ce{F2}",
           "H2O2": r"\ce{H2O2}", "NO2": r"\ce{NO2}", "N2O": r"\ce{N2O}"}

COORD_LABEL = {"bond": "bond length", "stretch": "symmetric stretch", "bend": "bend angle",
               "twist": "dihedral twist", "nn_stretch": "N--N stretch",
               "no_stretch": "N--O stretch", "surface": "2-D surface"}


def f(v, d=4, dash="--"):
    if v is None:
        return dash
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return r"n/a"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def g(v, d=3, dash="--"):
    """general format, strips trailing zeros"""
    if v is None:
        return dash
    if isinstance(v, float) and math.isnan(v):
        return "n/a"
    if isinstance(v, float):
        s = f"{v:.{d}f}".rstrip("0").rstrip(".")
        return s if s else "0"
    return str(v)


# ---------------------------------------------------------------- load scans
scans = []
for path in sorted(glob.glob(f"{REPO}/experiment/*/*_scan.json")):
    d = json.load(open(path))
    m, rows = d["metadata"], d["rows"]
    live = [r for r in rows if r.get("vqe") is not None]
    cand = [k for k in ("distance", "bond", "oo", "oh", "angle", "dihedral", "r_nn", "r_no")
            if k in live[0]]
    varying = [k for k in cand if len({round(r[k], 6) for r in live}) > 1]
    errs = [(r["vqe"] - r["exact"]) * 1000 for r in live if r.get("exact") is not None]
    hfe = [(r["hf"] - r["exact"]) * 1000 for r in live
           if r.get("exact") is not None and r.get("hf") is not None]
    best = min(live, key=lambda r: r["vqe"])
    nev = [r.get("n_evaluations") for r in live if r.get("n_evaluations")]
    scans.append(dict(path=path, name=os.path.basename(path), meta=m, rows=rows, live=live,
                      mol=m.get("molecule"), coord=m.get("coordinate", "bond"),
                      stype=m.get("scan_type", "curve"), varying=varying, best=best,
                      Emin=best["vqe"],
                      maxerr=max(abs(e) for e in errs) if errs else None,
                      meanerr=sum(abs(e) for e in errs) / len(errs) if errs else None,
                      maxhf=max(abs(e) for e in hfe) if hfe else None,
                      nev=sum(nev) // len(nev) if nev else None))

by_mol = defaultdict(list)
for s in scans:
    by_mol[s["mol"]].append(s)

# --------------------------------------------------- load interpolation runs
interp = []
for path in sorted(glob.glob(f"{REPO}/experiment/*_polynomial/*_interp.json")
                   + glob.glob(f"{REPO}/experiment/*_spline/*_interp.json")):
    d = json.load(open(path))
    interp.append(dict(path=path, name=os.path.basename(path), meta=d["metadata"],
                       rows=d.get("rows", [])))

imol = defaultdict(list)
for i in interp:
    imol[i["meta"]["molecule"]].append(i)


def interp_key(rec):
    """(coordinate-ish tag, family) for an interpolation record"""
    n = rec["name"]
    fam = "spline" if "spline" in n else "polynomial"
    stem = n.replace("_interp.json", "")
    for tag in ("surface", "no_stretch", "nn_stretch", "stretch", "bend", "twist"):
        if tag in stem:
            return tag, fam
    return "bond", fam


L = []
A = L.append

A(r"% ====================================================================")
A(r"%  tables.tex -- GENERATED from the repository's saved scan and")
A(r"%  interpolation files by gen_tables.py.  Do not edit by hand.")
A(r"% ====================================================================")
A("")

# ============================================================ TABLE: campaign
A(r"\begin{table}[!t]")
A(r"\caption{Complete computational campaign. Every VQE scan stored in the "
  r"repository, with the active space, qubit count and ansatz parameter count "
  r"actually used. $N$ is the number of geometries at which the variational "
  r"problem was solved. Bond lengths are in \AA{} and angles in degrees.}")
A(r"\label{tab:campaign}")
A(r"\centering\small")
A(r"\resizebox{\textwidth}{!}{%")
A(r"\begin{tabular}{llllrrrlll}")
A(r"\toprule")
A(r"Molecule & Coordinate & Type & Range / step & $N$ & $n_q$ & $p$ & "
  r"Active space & Optimizer & Basis \\")
A(r"\midrule")
for mol in MOL_ORDER:
    for s in sorted(by_mol[mol], key=lambda x: x["name"]):
        m = s["meta"]
        if s["stype"] == "surface":
            axes = [(k, v) for k, v in m.items() if k.endswith("_values")]
            shape = r"$\times$".join(str(len(v)) for _, v in axes)
            rng = f"{shape} grid"
        elif s["coord"] == "bend" and m.get("angle_start") is not None:
            rng = (f"{g(m.get('angle_start'))}--{g(m.get('angle_stop'))} / "
                   f"{g(m.get('angle_step'))}")
        else:
            rng = f"{g(m.get('start'))}--{g(m.get('stop'))} / {g(m.get('step'))}"
        AS = (f"({m.get('active_electrons')}e, {m.get('active_orbitals')}o)"
              if m.get("active_electrons") is not None else "full")
        opt = m.get("optimizer") or "default"
        A(f"{TEX_MOL[mol]} & {COORD_LABEL.get(s['coord'], s['coord'])} & "
          f"{s['stype']} & {rng} & {len(s['rows'])} & "
          f"{s['live'][0].get('n_qubits') or '--'} & "
          f"{s['live'][0].get('n_params') or '--'} & {AS} & {opt} & "
          f"{m.get('basis', 'sto-3g')} \\\\")
    A(r"\addlinespace[1pt]")
A(r"\bottomrule")
A(r"\end{tabular}}")
A(r"\end{table}")
A("")

# ======================================================= TABLE: VQE accuracy
A(r"\begin{table}[!t]")
A(r"\caption{Accuracy of the variational solution against exact "
  r"diagonalisation of the same active-space Hamiltonian, in milli-Hartree. "
  r"$\overline{\Delta}$ and $\Delta_{\max}$ are the mean and maximum of "
  r"$|E_{\mathrm{VQE}}-E_{\mathrm{exact}}|$ over the scan; "
  r"$\Delta^{\mathrm{HF}}_{\max}$ is the largest correlation energy the "
  r"Hartree--Fock reference misses on the same points, i.e.\ the error the "
  r"variational calculation is trying to recover. The chemical-accuracy "
  r"threshold is $1.6$~mHa.}")
A(r"\label{tab:vqeacc}")
A(r"\centering\small")
A(r"\resizebox{\textwidth}{!}{%")
A(r"\begin{tabular}{llrrrrrl}")
A(r"\toprule")
A(r"Molecule & Coordinate & $\overline{\Delta}$ & $\Delta_{\max}$ & "
  r"$\Delta^{\mathrm{HF}}_{\max}$ & $E_{\min}$ (Ha) & evals & Chem.\ acc.? \\")
A(r"\midrule")
for mol in MOL_ORDER:
    for s in sorted(by_mol[mol], key=lambda x: x["name"]):
        ok = (r"\textbf{yes}" if (s["meanerr"] is not None and s["meanerr"] < 1.6)
              else "no")
        A(f"{TEX_MOL[mol]} & {COORD_LABEL.get(s['coord'], s['coord'])} & "
          f"{f(s['meanerr'])} & {f(s['maxerr'])} & {f(s['maxhf'], 2)} & "
          f"{f(s['Emin'], 6)} & {s['nev'] or '--'} & {ok} \\\\")
    A(r"\addlinespace[1pt]")
A(r"\bottomrule")
A(r"\end{tabular}}")
A(r"\end{table}")
A("")

# ================================================ TABLE: interpolation master
A(r"\begin{table}[!t]")
A(r"\caption{Out-of-sample reconstruction error for every scan that was "
  r"post-processed, in milli-Hartree. One-dimensional scans are scored by "
  r"leave-one-out at the interior nodes; surfaces by the half-grid holdout of "
  r"Section~\ref{sec:holdout}. \emph{Linear} is the piecewise-linear baseline "
  r"implied by every raw scan plot. A dash means the method was not applied to "
  r"that scan; \emph{n/a} means the full interpolating degree left too few "
  r"points to score.}")
A(r"\label{tab:interp}")
A(r"\centering\small")
A(r"\resizebox{\textwidth}{!}{%")
A(r"\begin{tabular}{llrrrrrrrr}")
A(r"\toprule")
A(r" & & \multicolumn{2}{c}{Linear baseline} & \multicolumn{3}{c}{Global polynomial} "
  r"& \multicolumn{2}{c}{Cubic spline} & \\")
A(r"\cmidrule(lr){3-4}\cmidrule(lr){5-7}\cmidrule(lr){8-9}")
A(r"Molecule & Coordinate & RMS & max & best $d$ & RMS & worst RMS & RMS & max "
  r"& Winner \\")
A(r"\midrule")
for mol in MOL_ORDER:
    recs = defaultdict(dict)
    for r in imol.get(mol, []):
        tag, fam = interp_key(r)
        recs[tag][fam] = r["meta"]
    if not recs:
        continue
    for tag in ("bond", "stretch", "nn_stretch", "no_stretch", "bend", "twist", "surface"):
        if tag not in recs:
            continue
        p = recs[tag].get("polynomial", {})
        s = recs[tag].get("spline", {})
        lin_rms = p.get("linear_rms_mha", s.get("linear_rms_mha"))
        lin_max = p.get("linear_max_mha", s.get("linear_max_mha"))
        if tag == "surface":
            prms, pmax = p.get("polynomial_rms_mha"), p.get("polynomial_max_mha")
            bestd = p.get("featured_degree")
            worst = p.get("worst_degree_rms_mha")
        else:
            bestd = p.get("best_degree")
            prms = p.get("best_degree_rms_mha")
            worst = p.get("worst_degree_rms_mha")
        srms, smax = s.get("spline_rms_mha"), s.get("spline_max_mha")
        # decide winner
        cands = []
        if prms is not None and not (isinstance(prms, float) and math.isnan(prms)):
            cands.append(("poly", prms))
        if srms is not None and not (isinstance(srms, float) and math.isnan(srms)):
            cands.append(("spline", srms))
        if lin_rms is not None:
            cands.append(("linear", lin_rms))
        win = min(cands, key=lambda c: c[1])[0] if cands else "--"
        wintex = {"poly": "polynomial", "spline": "spline", "linear": "linear",
                  "--": "--"}[win]
        A(f"{TEX_MOL[mol]} & {COORD_LABEL.get(tag, tag)} & {f(lin_rms, 3)} & "
          f"{f(lin_max, 3)} & {bestd if bestd is not None else '--'} & "
          f"{f(prms, 3)} & {f(worst, 2)} & {f(srms, 3)} & {f(smax, 3)} & "
          f"{wintex} \\\\")
    A(r"\addlinespace[1pt]")
A(r"\bottomrule")
A(r"\end{tabular}}")
A(r"\end{table}")
A("")

# ==================================================== TABLE: geometry summary
A(r"\begin{table}[!t]")
A(r"\caption{Equilibrium geometries recovered from the reconstructed curves "
  r"and surfaces, compared with the best node actually computed and with the "
  r"reference value (experimental where quoted in the scan metadata, otherwise "
  r"the model's own optimised geometry). Lengths in \AA, angles in degrees. "
  r"The gap between the \emph{best node} and the \emph{interpolated} column is "
  r"the resolution the interpolant buys over the raw grid.}")
A(r"\label{tab:geom}")
A(r"\centering\small")
A(r"\resizebox{\textwidth}{!}{%")
A(r"\begin{tabular}{lllrrrr}")
A(r"\toprule")
A(r"Molecule & Coordinate & Method & Best node & Interpolated & Reference & "
  r"$|$Interp.$-$Ref.$|$ \\")
A(r"\midrule")
for mol in MOL_ORDER:
    for r in sorted(imol.get(mol, []), key=lambda x: x["name"]):
        m = r["meta"]
        tag, fam = interp_key(r)
        if tag == "surface":
            continue  # surfaces handled per-molecule
        node = m.get("best_node_x", m.get("best_node_bond", m.get("best_node")))
        imin = m.get("interpolated_min_x", m.get("interpolated_min_bond",
                                                 m.get("interpolated_min")))
        ref = (m.get("reference_x") or m.get("experimental_bond")
               or m.get("model_reference") or m.get("experimental_reference")
               or m.get("model_bond") or m.get("reference_bond"))
        if imin is None:
            continue
        dev = abs(imin - ref) if (ref is not None and imin is not None) else None
        A(f"{TEX_MOL[mol]} & {COORD_LABEL.get(tag, tag)} & {fam} & {g(node, 4)} & "
          f"{g(imin, 4)} & {g(ref, 4)} & {g(dev, 4)} \\\\")
    A(r"\addlinespace[1pt]")
A(r"\bottomrule")
A(r"\end{tabular}}")
A(r"\end{table}")
A("")

# ========================================== PER-SCAN FULL ENERGY DATA TABLES
AXIS_LABEL = {"distance": r"$r$ (\AA)", "bond": r"$r$ (\AA)", "oo": r"$r_{\mathrm{OO}}$ (\AA)",
              "oh": r"$r_{\mathrm{OH}}$ (\AA)", "angle": r"$\theta$ (deg)",
              "dihedral": r"$\phi$ (deg)", "r_nn": r"$r_{\mathrm{NN}}$ (\AA)",
              "r_no": r"$r_{\mathrm{NO}}$ (\AA)"}


def data_table(s, label, twocol=False, maxrows=None):
    m, rows = s["meta"], s["live"]
    key = s["varying"][0] if s["varying"] else "distance"
    out = []
    env = "table" if twocol else "table"
    out.append(rf"\begin{{{env}}}[!t]")
    cap = (rf"{TEX_MOL[s['mol']]} {COORD_LABEL.get(s['coord'], s['coord'])} scan: "
           rf"Hartree--Fock, VQE and exact energies (Ha) at every geometry; "
           rf"$\delta = (E_{{\mathrm{{VQE}}}}-E_{{\mathrm{{exact}}}})\times 10^3$ "
           rf"(mHa).")
    out.append(rf"\caption{{{cap}}}")
    out.append(rf"\label{{{label}}}")
    out.append(r"\centering\small")
    ncol = 2 if (maxrows and len(rows) > maxrows) else 1
    if ncol == 2:
        out.append(r"\begin{tabular}{rrrrr@{\hskip 1.2em}rrrrr}")
    else:
        out.append(r"\begin{tabular}{rrrrr}")
    out.append(r"\toprule")
    head = (AXIS_LABEL.get(key, key) + r" & $E_{\mathrm{HF}}$ & $E_{\mathrm{VQE}}$ "
            r"& $E_{\mathrm{exact}}$ & $\delta$")
    out.append((head + " & " + head + r" \\") if ncol == 2 else head + r" \\")
    out.append(r"\midrule")

    def cells(r):
        ex = r.get("exact")
        d = (r["vqe"] - ex) * 1000 if ex is not None else None
        return (f"{g(r[key], 4)} & {f(r.get('hf'), 6)} & {f(r['vqe'], 6)} & "
                f"{f(ex, 6)} & {f(d, 3)}")

    if ncol == 2:
        half = (len(rows) + 1) // 2
        for i in range(half):
            left = cells(rows[i])
            right = cells(rows[i + half]) if i + half < len(rows) else " & & & & "
            out.append(f"{left} & {right} \\\\")
    else:
        for r in rows:
            out.append(cells(r) + r" \\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    out.append(rf"\end{{{env}}}")
    out.append("")
    return out


for mol in MOL_ORDER:
    for s in sorted(by_mol[mol], key=lambda x: x["name"]):
        if s["stype"] == "surface":
            continue
        tag = s["coord"]
        lbl = f"tab:data:{mol.lower()}:{tag}"
        big = len(s["live"]) > 24
        L.extend(data_table(s, lbl, twocol=big, maxrows=24 if big else None))

# ================================================== DEGREE SWEEP TABLES
A(r"% ---- polynomial degree sweeps ----")
sweeps = defaultdict(list)
for path in sorted(glob.glob(f"{REPO}/experiment/*_polynomial/*_degrees.csv")):
    stem = os.path.basename(path).replace("_polynomial_degrees.csv", "")
    mol = stem.split("_")[0]
    sweeps[mol].append((stem, path))

MOLKEY = {m.lower(): m for m in MOL_ORDER}
for molkey in sorted(sweeps):
    mol = MOLKEY.get(molkey)
    if mol is None:
        continue
    for stem, path in sweeps[molkey]:
        rows = list(csv.DictReader(open(path)))
        tag = stem[len(molkey):].strip("_") or "bond"
        A(r"\begin{table}[!t]")
        A(rf"\caption{{{TEX_MOL[mol]} {COORD_LABEL.get(tag, tag)} polynomial "
          rf"degree sweep: leave-one-out error (mHa) against fitted degree "
          rf"$d$; the best degree is in bold.}}")
        A(rf"\label{{tab:deg:{molkey}:{tag}}}")
        A(r"\centering\small")
        A(r"\begin{tabular}{rrrr}")
        A(r"\toprule")
        A(r"Degree $d$ & max & RMS & MAE \\")
        A(r"\midrule")
        best = min(rows, key=lambda r: float(r["rms_mha"]))
        for r in rows:
            mark = r"\textbf{" if r is best else ""
            end = "}" if r is best else ""
            A(f"{mark}{r['degree']}{end} & {mark}{float(r['max_mha']):.4f}{end} & "
              f"{mark}{float(r['rms_mha']):.4f}{end} & "
              f"{mark}{float(r['mae_mha']):.4f}{end} \\\\")
        A(r"\bottomrule")
        A(r"\end{tabular}")
        A(r"\end{table}")
        A("")

# =============================================== TABLE: surface grid details
A(r"\begin{table}[!t]")
A(r"\caption{The five two-dimensional scans. $N$ is the total number of "
  r"variational solves, which grows as the product of the two axis lengths; "
  r"this quadratic cost is the reason the holdout experiment of "
  r"Section~\ref{sec:holdout} matters.}")
A(r"\label{tab:surfaces}")
A(r"\centering\small")
A(r"\begin{tabular}{llrr}")
A(r"\toprule")
A(r"Molecule & Grid (axis $\times$ axis) & $N$ & $n_q$ \\")
A(r"\midrule")
for mol in MOL_ORDER:
    for s in by_mol[mol]:
        if s["stype"] != "surface":
            continue
        m = s["meta"]
        axes = [(k.replace("_values", ""), v) for k, v in m.items() if k.endswith("_values")]
        desc = ", ".join(f"{k}: {len(v)} pts {g(min(v))}--{g(max(v))}" for k, v in axes)
        A(f"{TEX_MOL[mol]} & {desc} & {len(s['rows'])} & "
          f"{s['live'][0].get('n_qubits')} \\\\")
A(r"\bottomrule")
A(r"\end{tabular}")
A(r"\end{table}")
A("")

# ============================================ split + format for two columns
def to_acm(block, label):
    """Turn one old-style table block into an acmart-friendly one."""
    wide = label in WIDE
    env = "table*" if wide else "table"
    pos = "[!t]" if wide else "[!htbp]"
    width = r"\textwidth" if wide else r"\columnwidth"
    block = block.replace(r"\begin{table}[!t]", rf"\begin{{{env}}}{pos}", 1)
    block = block.replace(r"\end{table}", rf"\end{{{env}}}")
    # font size / column padding
    size = r"\small" if wide else r"\footnotesize"
    block = block.replace(r"\centering\small",
                          rf"\centering{size}\setlength{{\tabcolsep}}{{{'3.5pt' if wide else '3pt'}}}")
    # shrink-to-fit (never enlarge) instead of forcing \textwidth
    block = block.replace(r"\resizebox{\textwidth}{!}{%", "")
    block = block.replace(r"\end{tabular}}", r"\end{tabular}")
    block = block.replace(r"\begin{tabular}",
                          rf"\adjustbox{{max width={width}}}{{%" + "\n" + r"\begin{tabular}", 1)
    block = block.replace(r"\end{tabular}", r"\end{tabular}}", 1)
    if label == "tab:geom":
        block = split_halves(block)
    return block


def split_halves(block):
    """35 geometry rows on a page-wide table would be tall and thin: lay the
    rows out as two tabulars side by side instead."""
    head = re.search(r"\\toprule\n(.*?)\n\\midrule", block, re.S).group(1)
    body = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
    rows = [r for r in body.split("\n") if r.strip().endswith(r"\\")]
    half = (len(rows) + 1) // 2
    cols = re.search(r"\\begin\{tabular\}\{(\w+)\}", block).group(1)

    def tab(rs):
        return ("\\begin{tabular}{%s}\n\\toprule\n%s\n\\midrule\n%s\n\\bottomrule\n\\end{tabular}"
                % (cols, head, "\n".join(rs)))
    pre = block.split(r"\adjustbox", 1)[0]
    return (pre + r"\adjustbox{max width=\textwidth}{%" + "\n" + tab(rows[:half]) +
            r"\hspace{1.5em}" + tab(rows[half:]) + "}\n" + r"\end{table*}" + "\n")


text = "\n".join(L)
blocks = re.findall(r"\\begin\{table\}\[!t\].*?\\end\{table\}", text, flags=re.S)
os.makedirs(os.path.join(HERE, "tables"), exist_ok=True)
for old in glob.glob(os.path.join(HERE, "tables", "*.tex")):
    os.remove(old)
n = 0
for b in blocks:
    label = re.search(r"\\label\{([^}]*)\}", b).group(1)
    fn = os.path.join(HERE, "tables", label.replace(":", "_") + ".tex")
    with open(fn, "w", encoding="utf-8") as fh:
        fh.write(f"% GENERATED by gen_tables.py -- do not edit by hand ({label})\n")
        fh.write(to_acm(b, label) + "\n")
    n += 1
print(f"wrote {n} table files to {os.path.join(HERE, 'tables')}")
