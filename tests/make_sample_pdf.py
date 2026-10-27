"""Create sample_chapter.pdf from the built-in sample text (handy for testing PDF upload)."""
import io, os, sys, textwrap
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from ccart.sample_text import SAMPLE_CHAPTER

out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sample_chapter.pdf")
c = canvas.Canvas(out, pagesize=A4)
for n, pg in enumerate(SAMPLE_CHAPTER.split("\f"), 1):
    y = 800
    c.setFont("Helvetica", 9); c.drawString(60, 820, "NLP Textbook - Sample Chapter")
    c.setFont("Helvetica", 11)
    for para in pg.strip().split("\n\n"):
        for line in textwrap.wrap(para, 85):
            c.drawString(60, y, line); y -= 14
        y -= 10
    c.drawString(300, 30, str(n)); c.showPage()
c.save()
print("wrote", out)
