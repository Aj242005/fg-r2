"""
PDF Report Generation — Server-Side
======================================
Generates E-Challan and Audit Report PDFs using ReportLab.
"""

import io
import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_RIGHT


FINE_MAP = {
    "helmet": 1000,
    "no_helmet": 1000,
    "red_light": 500,
    "tripling": 1500,
    "modified": 2000,
    "stop_line": 500,
    "parking": 500,
}


def _get_styles():
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="ChallanTitle",
        fontName="Helvetica-Bold",
        fontSize=18,
        alignment=TA_CENTER,
        spaceAfter=4 * mm,
    ))
    styles.add(ParagraphStyle(
        name="ChallanSubtitle",
        fontName="Helvetica",
        fontSize=10,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceAfter=8 * mm,
    ))
    styles.add(ParagraphStyle(
        name="SectionHeader",
        fontName="Helvetica-Bold",
        fontSize=12,
        spaceBefore=6 * mm,
        spaceAfter=3 * mm,
    ))
    styles.add(ParagraphStyle(
        name="Disclaimer",
        fontName="Helvetica",
        fontSize=8,
        alignment=TA_CENTER,
        textColor=colors.grey,
        spaceBefore=10 * mm,
    ))
    return styles


def generate_challan_pdf(violation: dict) -> bytes:
    """Generate an E-Challan PDF for a single violation. Returns raw PDF bytes."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = _get_styles()
    elements = []

    # Header
    elements.append(Paragraph("TRAFFIC POLICE DEPARTMENT", styles["ChallanTitle"]))
    elements.append(Paragraph("Automated Traffic Enforcement System — E-Challan", styles["ChallanSubtitle"]))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=6 * mm))

    # Challan details table
    challan_data = [
        ["Challan No:", str(violation.get("id", "N/A"))],
        ["Date & Time:", violation.get("created_at", datetime.datetime.now().isoformat())],
        ["Location:", violation.get("location", "Camera Feed")],
        ["Vehicle No:", violation.get("plate_number", "UNKNOWN")],
        ["Violation Type:", str(violation.get("violation_type", "N/A")).upper()],
        ["Confidence:", f"{float(violation.get('confidence', 0)) * 100:.1f}%"],
        ["Severity:", violation.get("severity", "MEDIUM")],
        ["Fine Amount:", f"₹{violation.get('fine_amount', 1000):,}"],
        ["Payment Status:", violation.get("status", "Unpaid")],
    ]

    t = Table(challan_data, colWidths=[50 * mm, 120 * mm])
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LINEBELOW", (0, -1), (-1, -1), 1, colors.lightgrey),
    ]))
    elements.append(t)

    # Disclaimer
    elements.append(Spacer(1, 20 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=4 * mm))
    elements.append(Paragraph(
        "This is a computer-generated document and does not require a physical signature.<br/>"
        "Please pay the fine amount within 15 days to avoid further legal action.",
        styles["Disclaimer"],
    ))

    doc.build(elements)
    return buf.getvalue()


def generate_audit_report_pdf(plate_number: str, violations: list[dict]) -> bytes:
    """Generate a multi-violation audit report PDF for a vehicle entity."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, topMargin=20 * mm, bottomMargin=20 * mm)
    styles = _get_styles()
    elements = []

    # Header
    elements.append(Paragraph("TRAFFIC AUDIT REPORT", styles["ChallanTitle"]))
    elements.append(Paragraph(
        f"Vehicle: {plate_number} | Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        styles["ChallanSubtitle"],
    ))
    elements.append(HRFlowable(width="100%", thickness=1, color=colors.black, spaceAfter=6 * mm))

    # Summary
    total_fine = sum(v.get("fine_amount", 0) for v in violations)
    unpaid = sum(v.get("fine_amount", 0) for v in violations if v.get("status") == "Unpaid")
    elements.append(Paragraph("Summary", styles["SectionHeader"]))
    summary_data = [
        ["Total Violations:", str(len(violations))],
        ["Total Fines:", f"₹{total_fine:,}"],
        ["Unpaid:", f"₹{unpaid:,}"],
    ]
    st = Table(summary_data, colWidths=[50 * mm, 120 * mm])
    st.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 11),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
    ]))
    elements.append(st)

    # Violations table
    elements.append(Paragraph("Violation History", styles["SectionHeader"]))
    table_data = [["#", "Date", "Type", "Confidence", "Fine", "Status"]]
    for i, v in enumerate(violations, 1):
        created = v.get("created_at", "")
        if isinstance(created, str) and "T" in created:
            created = created.split("T")[0]
        table_data.append([
            str(i),
            str(created),
            str(v.get("violation_type", "")).upper(),
            f"{float(v.get('confidence', 0)) * 100:.1f}%",
            f"₹{v.get('fine_amount', 0):,}",
            v.get("status", "Unpaid"),
        ])

    vt = Table(table_data, colWidths=[10 * mm, 30 * mm, 35 * mm, 25 * mm, 25 * mm, 25 * mm])
    vt.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a1a1a")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3 * mm),
        ("TOPPADDING", (0, 0), (-1, -1), 2 * mm),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f5")]),
    ]))
    elements.append(vt)

    # Disclaimer
    elements.append(Spacer(1, 15 * mm))
    elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey, spaceAfter=4 * mm))
    elements.append(Paragraph(
        "This report is auto-generated by the Automated Traffic Enforcement System.<br/>"
        "For disputes, contact the Traffic Police Department within 30 days of issue.",
        styles["Disclaimer"],
    ))

    doc.build(elements)
    return buf.getvalue()
