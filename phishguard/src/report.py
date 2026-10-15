"""
PhishGuard :: comparison report (v3)
--------------------------------------
Builds the final-step report that positions a PhishGuard result against the
way phishing has traditionally been detected, using the team's literature
survey as the source for the "traditional" side.

Two flavours share one data structure:

  * per-scan report   - build_report(scan)  : this URL's verdict, how the
                        baseline models would have judged the SAME URL, how
                        much the fetched HTML changed the score, then the
                        project-level sections below.
  * project report    - build_report(None)  : capability comparison vs the
                        surveyed approaches, the six research gaps and how
                        PhishGuard responds, same-data benchmark, reported
                        literature accuracies (clearly labelled as not
                        directly comparable), and limitations.

build_report() returns plain dicts/lists so the same content feeds the HTML
page (Jinja) and the PDF (reportlab, render_pdf()).
"""

import csv
import io
import json
import os
from datetime import datetime
from xml.sax.saxutils import escape

from . import literature_data as lit

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")


# ----------------------------- data assembly --------------------------------

def _load_baseline_benchmark():
    path = os.path.join(RESULTS_DIR, "baseline_comparison.csv")
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows


def _load_metrics_meta():
    path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _agreement_note(phishguard_verdict, baselines):
    if not baselines:
        return ""
    binary = [b["verdict"] for b in baselines]
    n_phish = binary.count("PHISHING")
    n = len(binary)
    if phishguard_verdict == "SUSPICIOUS":
        return (f"PhishGuard rated this URL SUSPICIOUS (an uncertain band that binary baselines "
                f"cannot express); {n_phish} of {n} baselines said PHISHING.")
    pg_is_phish = phishguard_verdict == "PHISHING"
    agree = n_phish if pg_is_phish else n - n_phish
    if agree == n:
        return f"All {n} baseline models reached the same verdict as PhishGuard."
    return (f"{agree} of {n} baseline models agreed with PhishGuard's verdict; "
            f"the baselines disagree with each other, which a single static classifier would hide.")


def build_report(scan=None):
    """scan: the dict returned by app.score(), or None for the project report."""
    bench = _load_baseline_benchmark()
    meta = _load_metrics_meta()
    report = {
        "generated_at": datetime.now().strftime("%d %b %Y, %H:%M"),
        "survey_title": lit.SURVEY_TITLE,
        "scan": None,
        "capability": {
            "columns": lit.CAPABILITY_COLUMNS,
            "rows": lit.CAPABILITY_ROWS + [lit.PHISHGUARD_ROW],
            "footnote": lit.CAPABILITY_FOOTNOTE,
        },
        "gaps": lit.GAPS,
        "benchmark": {
            "rows": bench,
            "note": ("Measured by PhishGuard: every model trained and tested on the same "
                     "synthetic dataset and the same 80/20 split."),
        },
        "reported_accuracy": {
            "rows": lit.REPORTED_ACCURACY,
            "caveat": lit.ACCURACY_CAVEAT,
        },
        "limitations": lit.LIMITATIONS,
    }
    if scan:
        baselines = scan.get("baselines", [])
        report["scan"] = {
            "url": scan["url"],
            "verdict": scan["verdict"],
            "probability": scan["phishing_probability"],
            "confidence": scan["confidence"],
            "rationale": scan["rationale"],
            "top_features": scan["top_features"][:5],
            "used_html": scan["used_html"],
            "fetch": scan.get("fetch", {}),
            "baselines": baselines,
            "agreement": _agreement_note(scan["verdict"], baselines),
            "signal": scan.get("signal_comparison"),
        }
    return report


# ----------------------------- PDF rendering --------------------------------

def render_pdf(report: dict) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle, KeepTogether,
    )

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=9.5, leading=13, alignment=TA_LEFT)
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10.5,
                           textColor=colors.HexColor("#444444"))
    cell = ParagraphStyle("cell", parent=body, fontSize=8, leading=10)
    cell_b = ParagraphStyle("cellb", parent=cell, fontName="Helvetica-Bold")
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontName="Helvetica-Bold",
                        fontSize=17, leading=21, spaceAfter=4)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=12, leading=15, spaceBefore=12, spaceAfter=5,
                        textColor=colors.HexColor("#1f3b73"))

    def P(text, style=body):
        return Paragraph(escape(str(text)), style)

    def table(header, rows, widths, align_center_cols=()):
        data = [[Paragraph(escape(h), cell_b) for h in header]]
        for r in rows:
            data.append([Paragraph(escape(str(c)).replace("\n", "<br/>"), cell) for c in r])
        t = Table(data, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#999999")),
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eefb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return t

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm,
                            title="PhishGuard comparison report", author="PhishGuard")
    s = []
    scan = report.get("scan")
    s.append(P("PhishGuard comparison report", h1))
    s.append(P(("Scan report: " if scan else "Project report: ") + report["generated_at"], small))
    s.append(P("Positions PhishGuard against the approaches reviewed in: " + report["survey_title"], small))
    s.append(Spacer(1, 6))

    if scan:
        s.append(P("1. This scan", h2))
        s.append(table(["Field", "Value"], [
            ["URL", scan["url"]],
            ["Verdict", f'{scan["verdict"]}  (phishing probability {scan["probability"]:.1%}, confidence {scan["confidence"]:.1%})'],
            ["Rationale", scan["rationale"]],
            ["Page HTML used", ("Yes, fetched automatically" if scan["used_html"]
                                else "No, scored from the URL alone")],
        ], [38, 140]))
        fetch = scan.get("fetch") or {}
        if fetch.get("message"):
            s.append(Spacer(1, 3)); s.append(P("Fetch: " + fetch["message"], small))
        if scan["top_features"]:
            s.append(Spacer(1, 5))
            s.append(table(["Top contributing feature", "Value", "Contribution"],
                           [[f["feature"], f["value"], f'{f["contribution"]:+.4f}'] for f in scan["top_features"]],
                           [90, 40, 48]))

        s.append(P("2. How traditional models judge the same URL", h2))
        s.append(P("The four baselines below stand for the usual single static classifier with a "
                   "binary 0.5 verdict and no explanation. They were trained on the same data and split as PhishGuard.", small))
        s.append(Spacer(1, 3))
        rows = [["PhishGuard (RandomForest, calibrated, 3-band)", f'{scan["probability"]:.1%}',
                 scan["verdict"], "Yes, per-feature"]]
        for b in scan["baselines"]:
            rows.append([b["model"], f'{b["probability"]:.1%}', b["verdict"], "No"])
        s.append(table(["Model", "Phishing probability", "Verdict", "Explains why"], rows, [78, 38, 32, 30]))
        if scan.get("agreement"):
            s.append(Spacer(1, 3)); s.append(P(scan["agreement"], small))

        sig = scan.get("signal")
        if sig:
            s.append(P("3. What the fetched page added", h2))
            s.append(table(["Signal used", "Phishing probability", "Verdict"], [
                ["URL only", f'{sig["url_only_probability"]:.1%}', sig["url_only_verdict"]],
                ["URL + page HTML", f'{sig["hybrid_probability"]:.1%}', sig["hybrid_verdict"]],
            ], [70, 50, 40]))
            s.append(Spacer(1, 3)); s.append(P(sig["note"], small))
            n = 4
        else:
            n = 3
    else:
        n = 1

    cap = report["capability"]
    s.append(P(f"{n}. Capability comparison with surveyed approaches", h2))
    rows = [[r["approach"] + "\n" + r["survey_category"]] + r["cells"] for r in cap["rows"]]
    s.append(table(["Approach"] + cap["columns"], rows, [48, 24, 26, 26, 28, 26]))
    s.append(Spacer(1, 3)); s.append(P(cap["footnote"], small))
    s.append(Spacer(1, 4))
    s.append(table(["Approach", "Basis in the survey"],
                   [[r["approach"], r["basis"]] for r in cap["rows"]], [48, 130]))
    n += 1

    s.append(P(f"{n}. Research gaps from the survey and PhishGuard's response", h2))
    s.append(table(["Gap", "Evidence in surveyed work", "PhishGuard response", "Status"],
                   [[g["gap"], g["evidence"], g["response"], g["status"]] for g in report["gaps"]],
                   [34, 55, 55, 34]))
    n += 1

    bm = report["benchmark"]
    block = [P(f"{n}. Same-data benchmark (measured by PhishGuard)", h2)]
    if bm["rows"]:
        block.append(table(["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "FPR"],
                           [[r["model"], r["accuracy"], r["precision"], r["recall"], r["f1"],
                             r["roc_auc"], r["fpr"]] for r in bm["rows"]],
                           [52, 20, 20, 20, 20, 22, 20]))
    block.append(Spacer(1, 3)); block.append(P(bm["note"], small))
    s.append(KeepTogether(block))
    n += 1

    ra = report["reported_accuracy"]
    s.append(P(f"{n}. Accuracy reported in the literature (not directly comparable)", h2))
    s.append(table(["Category", "Method", "Ref.", "Reported accuracy"],
                   [list(r) for r in ra["rows"]], [34, 86, 16, 42]))
    s.append(Spacer(1, 3)); s.append(P(ra["caveat"], small))
    n += 1

    s.append(P(f"{n}. Limitations", h2))
    for title, text in report["limitations"]:
        s.append(Paragraph(f"<b>{escape(title)}.</b> {escape(text)}", body))
        s.append(Spacer(1, 3))

    doc.build(s)
    return buf.getvalue()
