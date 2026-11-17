"""Generates the sample eval corpus (a fictional company's documents) into evals/corpus/.

Run:  uv run python -m evals.make_corpus

- handbook.pdf: 6-page employee handbook (text layer)
- invoice_kl_0147.pdf: 2-page invoice (text layer)
- office_notice_scan.pdf: an image-only page, so answering from it requires OCR"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer

OUT = Path(__file__).parent / "corpus"

HANDBOOK = [
    (
        "1. Leave Policy",
        [
            "Every full-time employee of Kestrel Labs Pvt Ltd receives 24 days of paid annual leave per calendar year.",
            "In addition, employees get 12 days of sick leave per year. A medical certificate is required for sick "
            "leave longer than 2 consecutive days.",
            "Up to 10 unused annual leave days can be carried forward to the next year; any remaining days lapse on "
            "31 December.",
            "Leave of more than 3 consecutive days must be applied for at least 7 days in advance through the HR "
            "portal.",
            "Maternity leave is 26 weeks of fully paid leave. Paternity leave is 10 working days, to be taken within "
            "3 months of the child's birth.",
        ],
    ),
    (
        "2. Working Hours and Remote Work",
        [
            "Core collaboration hours are 11:00 to 16:00 IST, Monday to Friday. Outside core hours, employees may "
            "plan their day flexibly.",
            "Employees may work from home up to 3 days per week, with their manager's approval.",
            "Employees who work from home receive an internet reimbursement of Rs 1,500 per month.",
            "The Kestrel Labs office is located on the 14th floor of Tower 2, GIFT City, Gandhinagar.",
        ],
    ),
    (
        "3. Travel and Expenses",
        [
            "Flights shorter than 6 hours must be booked in economy class. Business class is allowed only for flights "
            "longer than 6 hours, with director approval.",
            "Hotel stays are capped at Rs 6,000 per night in metro cities and Rs 4,000 per night in other cities.",
            "The daily meal allowance during business travel is Rs 1,200.",
            "Expense claims must be submitted within 30 days of the expense. Original receipts are required for any "
            "single expense above Rs 500.",
        ],
    ),
    (
        "4. Information Security",
        [
            "Passwords must be changed every 90 days. Multi-factor authentication (MFA) is mandatory for all company "
            "accounts.",
            "All laptops must have full-disk encryption enabled (FileVault on macOS, BitLocker on Windows).",
            "Security incidents, including lost devices, must be reported within 2 hours to security@kestrel.example.",
            "Customer data must never be stored on personal devices or personal cloud accounts.",
        ],
    ),
    (
        "5. Performance and Compensation",
        [
            "Performance reviews take place twice a year, in April and October.",
            "Salaries are credited on the last working day of every month.",
            "The notice period is 60 days for employees and 90 days for people managers.",
            "The employee referral bonus is Rs 50,000, paid after the referred candidate completes 90 days with the "
            "company.",
        ],
    ),
    (
        "6. Learning and Development",
        [
            "Each employee has a learning budget of Rs 40,000 per year for courses, books and certifications.",
            "Employees may attend up to 2 external conferences per year, sponsored by the company.",
            "Certification exam fees are reimbursed only after the employee passes the exam.",
        ],
    ),
]

INVOICE_PAGES = [
    [
        "TAX INVOICE",
        "Invoice number: KL-2026-0147",
        "Invoice date: 12 August 2026",
        "From: Kestrel Labs Pvt Ltd, Tower 2, GIFT City, Gandhinagar. GSTIN 24AABCK1234F1Z5",
        "Bill to: Orion Retail Ltd, Ahmedabad",
        "Line items:",
        "1. RAG pilot development: 120 hours at Rs 2,500 per hour = Rs 3,00,000",
        "2. Cloud hosting (August 2026) = Rs 45,000",
        "Subtotal: Rs 3,45,000",
        "GST at 18%: Rs 62,100",
        "Total amount due: Rs 4,07,100",
    ],
    [
        "Payment terms: Net 30. Payment is due by 11 September 2026.",
        "Late payments incur interest of 1.5% per month on the outstanding amount.",
        "Bank: HDFC Bank, account number ending 4821, IFSC HDFC0001234.",
        "For billing questions contact accounts@kestrel.example.",
    ],
]

NOTICE_LINES = [
    "KESTREL LABS - OFFICE NOTICE",
    "",
    "The GIFT City office will be closed on",
    "2 October 2026 for Gandhi Jayanti.",
    "",
    "Server maintenance window:",
    "Saturday 10 October 2026, 22:00 to 02:00 IST.",
    "Internal tools will be unavailable during this window.",
]


def make_handbook(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, (title, paragraphs) in enumerate(HANDBOOK):
        story += [Paragraph(f"Kestrel Labs Employee Handbook 2026 - {title}", styles["Heading1"]), Spacer(1, 12)]
        story += [Paragraph(p, styles["BodyText"]) for p in paragraphs]
        if i < len(HANDBOOK) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_invoice(path: Path) -> None:
    styles = getSampleStyleSheet()
    story = []
    for i, lines in enumerate(INVOICE_PAGES):
        story += [Paragraph(line, styles["BodyText"]) for line in lines]
        if i < len(INVOICE_PAGES) - 1:
            story.append(PageBreak())
    SimpleDocTemplate(str(path), pagesize=A4).build(story)


def make_scanned_notice(path: Path) -> None:
    """Draws the text into an image and saves it as an image-only PDF (no text layer)."""
    img = Image.new("RGB", (1654, 1100), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default(size=44)
    except TypeError:  # very old Pillow
        font = ImageFont.load_default()
    y = 100
    for line in NOTICE_LINES:
        draw.text((100, y), line, fill="black", font=font)
        y += 80
    img.save(path, format="PDF", resolution=200)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    make_handbook(OUT / "handbook.pdf")
    make_invoice(OUT / "invoice_kl_0147.pdf")
    make_scanned_notice(OUT / "office_notice_scan.pdf")
    print(f"Wrote corpus to {OUT}")


if __name__ == "__main__":
    main()
