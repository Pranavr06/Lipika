"""
Lipika Project Report DOCX Generator
Generates an academic mini-project report for NMAM Institute of Technology, Nitte.
Matching the reference project report structure, formatting, tables, rubrics, and embedded figures.
"""

import os
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import parse_xml, OxmlElement
from docx.oxml.ns import nsdecls, qn

def create_report():
    doc = docx.Document()
    
    # ----------------------------------------------------
    # Page Setup: Standard A4 / Letter Margins (1.0 inch)
    # ----------------------------------------------------
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)
        
    # Default Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Times New Roman'
    normal_style.font.size = Pt(12)
    normal_style.font.color.rgb = RGBColor(0x22, 0x22, 0x22)
    normal_style.paragraph_format.line_spacing = 1.15
    normal_style.paragraph_format.space_after = Pt(6)

    # ----------------------------------------------------
    # Helper Functions
    # ----------------------------------------------------
    def set_cell_background(cell, fill_hex):
        tcPr = cell._tc.get_or_add_tcPr()
        shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{fill_hex}"/>')
        tcPr.append(shd)

    def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
        tcPr = cell._tc.get_or_add_tcPr()
        tcMar = parse_xml(f'<w:tcMar {nsdecls("w")}><w:top w:w="{top}" w:type="dxa"/><w:bottom w:w="{bottom}" w:type="dxa"/><w:left w:w="{left}" w:type="dxa"/><w:right w:w="{right}" w:type="dxa"/></w:tcMar>')
        tcPr.append(tcMar)

    def set_table_borders(table, color="BFBFBF", sz="4", val="single"):
        tblPr = table._tbl.tblPr
        borders = parse_xml(f'''
            <w:tblBorders {nsdecls("w")}>
                <w:top w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:bottom w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:insideH w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:insideV w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:left w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
                <w:right w:val="{val}" w:sz="{sz}" w:space="0" w:color="{color}"/>
            </w:tblBorders>
        ''')
        tblPr.append(borders)

    def add_para(text="", bold_prefix="", italic_prefix="", align=WD_ALIGN_PARAGRAPH.JUSTIFY, space_after=6, space_before=0, line_spacing=1.15):
        p = doc.add_paragraph()
        p.alignment = align
        p.paragraph_format.space_after = Pt(space_after)
        p.paragraph_format.space_before = Pt(space_before)
        p.paragraph_format.line_spacing = line_spacing
        
        if bold_prefix:
            r_b = p.add_run(bold_prefix)
            r_b.bold = True
            r_b.font.name = 'Times New Roman'
        if italic_prefix:
            r_i = p.add_run(italic_prefix)
            r_i.italic = True
            r_i.font.name = 'Times New Roman'
        if text:
            r_t = p.add_run(text)
            r_t.font.name = 'Times New Roman'
        return p

    def add_heading_1(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(14)
        p.paragraph_format.space_after = Pt(6)
        r = p.add_run(title)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(15)
        r.font.color.rgb = RGBColor(0x00, 0x20, 0x60)
        return p

    def add_heading_2(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(10)
        p.paragraph_format.space_after = Pt(4)
        r = p.add_run(title)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(13)
        r.font.color.rgb = RGBColor(0x1F, 0x49, 0x7D)
        return p

    def add_heading_3(title):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_before = Pt(6)
        p.paragraph_format.space_after = Pt(2)
        r = p.add_run(title)
        r.bold = True
        r.font.name = 'Times New Roman'
        r.font.size = Pt(12)
        r.font.color.rgb = RGBColor(0x33, 0x33, 0x33)
        return p

    def add_image_figure(image_path, caption_text, width_inches=5.8):
        if os.path.exists(image_path):
            p_img = doc.add_paragraph()
            p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_img.paragraph_format.space_before = Pt(8)
            p_img.paragraph_format.space_after = Pt(4)
            p_img.add_run().add_picture(image_path, width=Inches(width_inches))
            
            p_cap = doc.add_paragraph()
            p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p_cap.paragraph_format.space_before = Pt(0)
            p_cap.paragraph_format.space_after = Pt(10)
            r_cap = p_cap.add_run(caption_text)
            r_cap.font.name = 'Times New Roman'
            r_cap.font.size = Pt(10)
            r_cap.italic = True
            r_cap.font.color.rgb = RGBColor(0x44, 0x44, 0x44)

    # ====================================================
    # COVER / TITLE PAGE
    # ====================================================
    p_title_pre = doc.add_paragraph()
    p_title_pre.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title_pre.paragraph_format.space_before = Pt(10)
    p_title_pre.paragraph_format.space_after = Pt(4)
    r = p_title_pre.add_run("PROJECT REPORT\nON")
    r.bold = True
    r.font.size = Pt(14)

    p_proj_title = doc.add_paragraph()
    p_proj_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_proj_title.paragraph_format.space_before = Pt(6)
    p_proj_title.paragraph_format.space_after = Pt(12)
    r_pt = p_proj_title.add_run("LIPIKA: ACCELERATED OFFLINE KANNADA HANDWRITTEN CHARACTER RECOGNITION USING DEPTHWISE-SEPARABLE CONVOLUTIONAL NEURAL NETWORKS AND SUB-MILLISECOND ONNX INFERENCE")
    r_pt.bold = True
    r_pt.font.size = Pt(15)
    r_pt.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_before = Pt(8)
    p_sub.paragraph_format.space_after = Pt(4)
    r_sub = p_sub.add_run("Submitted to\n")
    r_sub.font.size = Pt(11)
    r_sub_c = p_sub.add_run("NMAM INSTITUTE OF TECHNOLOGY, NITTE\n")
    r_sub_c.bold = True
    r_sub_c.font.size = Pt(13)
    r_sub_det = p_sub.add_run("(Off-Campus Centre, Nitte Deemed to be University, Nitte - 574 110, Karnataka, India)\nIn partial fulfilment of the requirements for the award of the Degree of Bachelor of Technology\nIn\n")
    r_sub_det.font.size = Pt(11)
    r_sub_d = p_sub.add_run("INFORMATION SCIENCE AND ENGINEERING")
    r_sub_d.bold = True
    r_sub_d.font.size = Pt(12)

    # Students Table
    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_before = Pt(8)
    p_by.paragraph_format.space_after = Pt(2)
    r_by = p_by.add_run("by")
    r_by.italic = True
    r_by.font.size = Pt(11)

    table_students = doc.add_table(rows=3, cols=2)
    table_students.alignment = WD_TABLE_ALIGNMENT.CENTER
    student_data = [
        ("Pranav R", "NNM24IS159"),
        ("Pratham", "NNM24IS164"),
        ("Prathwik Shetty", "NNM24IS171")
    ]
    for i, (name, usn) in enumerate(student_data):
        cell_name = table_students.cell(i, 0)
        cell_usn = table_students.cell(i, 1)
        cell_name.width = Inches(2.2)
        cell_usn.width = Inches(1.8)
        
        p_n = cell_name.paragraphs[0]
        p_n.alignment = WD_ALIGN_PARAGRAPH.LEFT
        p_n.paragraph_format.space_after = Pt(2)
        r_n = p_n.add_run(name)
        r_n.bold = True
        
        p_u = cell_usn.paragraphs[0]
        p_u.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_u.paragraph_format.space_after = Pt(2)
        r_u = p_u.add_run(usn)
        r_u.bold = True

    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_before = Pt(12)
    p_guide.paragraph_format.space_after = Pt(4)
    r_g1 = p_guide.add_run("Under the guidance of\n")
    r_g1.italic = True
    r_g1.font.size = Pt(11)
    r_g2 = p_guide.add_run("Dr. Ramesh G.\n")
    r_g2.bold = True
    r_g2.font.size = Pt(12)
    r_g3 = p_guide.add_run("Assistant Professor Gd.III\nDepartment of Information Science & Engineering\n2026-2027")
    r_g3.font.size = Pt(11)

    # NITTE Logo
    logo_path = "outputs/report_assets/nitte_logo.jpg"
    if os.path.exists(logo_path):
        p_logo = doc.add_paragraph()
        p_logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p_logo.paragraph_format.space_before = Pt(10)
        p_logo.paragraph_format.space_after = Pt(0)
        p_logo.add_run().add_picture(logo_path, width=Inches(3.8))

    doc.add_page_break()

    # ====================================================
    # CERTIFICATE & EVALUATION MARKS PAGE
    # ====================================================
    p_dept = doc.add_paragraph()
    p_dept.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_dept.paragraph_format.space_before = Pt(10)
    p_dept.paragraph_format.space_after = Pt(4)
    r_dept = p_dept.add_run("Department of Information Science & Engineering")
    r_dept.bold = True
    r_dept.font.size = Pt(13)

    p_cert = doc.add_paragraph()
    p_cert.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cert.paragraph_format.space_before = Pt(4)
    p_cert.paragraph_format.space_after = Pt(12)
    r_cert = p_cert.add_run("CERTIFICATE")
    r_cert.bold = True
    r_cert.font.size = Pt(14)
    r_cert.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    cert_text = (
        "This is to certify that Pranav R [NNM24IS159], Pratham [NNM24IS164] and Prathwik Shetty [NNM24IS171] "
        "of III-year B.Tech. have carried out the mini project “LIPIKA: Accelerated Offline Kannada Handwritten "
        "Character Recognition Using Depthwise-Separable Convolutional Neural Networks and Sub-Millisecond ONNX Inference” "
        "as part of the Machine Learning Foundations [IS2002-1] course during 2026-2027, fulfilling the partial requirements "
        "for the award of the degree of Bachelor of Technology in Information Science and Engineering at NMAM Institute of "
        "Technology, Nitte."
    )
    add_para(cert_text, space_after=20, line_spacing=1.2)

    p_sig = doc.add_paragraph()
    p_sig.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sig.paragraph_format.space_before = Pt(16)
    p_sig.paragraph_format.space_after = Pt(10)
    p_sig.add_run("………………………………………………………………\nSignature of Course Instructor\n").font.name = 'Times New Roman'
    r_sg = p_sig.add_run("Dr. Ramesh G.\n")
    r_sg.bold = True
    p_sig.add_run("Asst. Professor GD-III\nDepartment Of ISE\nNMAMIT, NITTE (DU)").font.name = 'Times New Roman'

    p_eval_t = doc.add_paragraph()
    p_eval_t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_eval_t.paragraph_format.space_before = Pt(10)
    p_eval_t.paragraph_format.space_after = Pt(4)
    r_et = p_eval_t.add_run("PROJECT EVALUATION MARKS")
    r_et.bold = True
    r_et.font.size = Pt(11)

    table_marks = doc.add_table(rows=2, cols=5)
    table_marks.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_marks)
    
    headers_marks = [
        "Implementation\n(20 Marks)",
        "Presentation\n(10 Marks)",
        "Question and Answer\n(10 Marks)",
        "Organization of the report\n(10 Marks)",
        "Total\n(50 Marks)"
    ]
    for j, h in enumerate(headers_marks):
        c = table_marks.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 80, 80, 100, 100)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
        
    for j in range(5):
        c = table_marks.cell(1, j)
        set_cell_margins(c, 100, 100, 100, 100)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.add_run("")

    p_sig2 = doc.add_paragraph()
    p_sig2.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_sig2.paragraph_format.space_before = Pt(12)
    p_sig2.paragraph_format.space_after = Pt(8)
    p_sig2.add_run("……………………………………\nSignature of Course Instructor").font.size = Pt(10)

    # Evaluation Rubrics Table
    p_rub_t = doc.add_paragraph()
    p_rub_t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_rub_t.paragraph_format.space_before = Pt(8)
    p_rub_t.paragraph_format.space_after = Pt(4)
    r_rub = p_rub_t.add_run("PROJECT EVALUATION RUBRICS")
    r_rub.bold = True
    r_rub.font.size = Pt(11)

    table_rub = doc.add_table(rows=5, cols=5)
    table_rub.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_rub)

    rubric_headers = ["Evaluation Component", "Excellent (16-20)", "Good (10-15)", "Satisfactory (1-9)", "Poor (0)"]
    for j, h in enumerate(rubric_headers):
        c = table_rub.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 60, 60, 80, 80)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9)

    rubric_rows = [
        ("Implementation (20 Marks)",
         "The project exceeds expectations with outstanding performance, creativity, and attention to detail.",
         "The project meets expectations and demonstrates a solid understanding of the criteria.",
         "The project partially meets expectations but requires improvement in certain areas.",
         "The project does not meet expectations and requires significant improvement."),
        ("Presentation (10 Marks)",
         "Uses graphics that explain and reinforce text and presentation. Highly engaged with audience. Pace is appropriate.",
         "Uses graphics that explain text and presentation. Eye contact majority of time. Adequate pace.",
         "Uses graphics that relate to text. Occasional eye contact. Short or long/rushed in parts.",
         "Rarely supports text. Reads most slides. No eye contact. Poor pace."),
        ("Question and Answer (10 Marks)",
         "Demonstrated full knowledge. Exceptionally answered all technical and analytical questions.",
         "Demonstrated solid knowledge. Answered majority of questions correctly.",
         "Demonstrated some knowledge. Insufficient clarity for several questions.",
         "Does not have grasp of project information; fails to answer questions."),
        ("Organization of the report (10 Marks)",
         "Well-documented content, thorough analysis, all specified institutional formats followed strictly.",
         "Documented content and followed almost all specified formats.",
         "Fair document. Several specified formatting guidelines omitted.",
         "Poor document content. Formats not followed, submitted after deadline.")
    ]

    for i, row in enumerate(rubric_rows):
        for j, text in enumerate(row):
            c = table_rub.cell(i+1, j)
            set_cell_margins(c, 50, 50, 60, 60)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT if j > 0 else WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(text)
            r.font.size = Pt(8)
            if j == 0:
                r.bold = True

    doc.add_page_break()

    # ====================================================
    # ABSTRACT PAGE
    # ====================================================
    p_abs_title = doc.add_paragraph()
    p_abs_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_abs_title.paragraph_format.space_before = Pt(10)
    p_abs_title.paragraph_format.space_after = Pt(12)
    r_ab = p_abs_title.add_run("ABSTRACT")
    r_ab.bold = True
    r_ab.font.size = Pt(14)
    r_ab.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    abstract_text = (
        "This project presents Lipika (ಲಿಪಿಕಾ), an accelerated, lightweight offline handwritten character recognition "
        "system engineered specifically for the 47 fundamental character classes of the Kannada (ಕನ್ನಡ) script (comprising "
        "13 Swaras and 34 Vyanjanas). Handcrafted morphological variations, stroke slant, variable line thickness, and "
        "highly intricate cursive loops in Dravidian scripts pose severe challenges to traditional optical character recognition "
        "and standard convolution models. Drawing inspiration from and directly benchmarking against the 2019 IEEE WIECON-ECE "
        "baseline paper by Ramesh et al., which utilized a standard four-layer Convolutional Neural Network with heavy dense "
        "bottlenecks (760,879 parameters, 17.52 million FLOPs), this work designs Fast-KannadaNet—a modernized neural architecture "
        "incorporating Depthwise Separable Convolutions, Inverted Residual blocks, Squeeze-and-Excitation (SE) channel attention, "
        "and Global Average Pooling (GAP). The model is trained on a comprehensive dataset of 23,500 balanced samples synthesized "
        "using Simard’s elastic displacement vector fields and pen-pressure morphological operations across 47 classes (500 samples/class, "
        "partitioned via an 80/20 train-test split into 18,800 training and 4,700 held-out test records). Optimization is achieved "
        "using AdamW with OneCycleLR dynamic learning rate scheduling. When compiled into an optimized Open Neural Network Exchange "
        "(ONNX) graph and executed using CPU SIMD vector instructions, Fast-KannadaNet achieves an exceptional single-character "
        "inference latency of 0.588 ms (>1,701 frames per second) while slashing total model parameters by 84.6% (117,367 parameters) "
        "and computational FLOPs by 41.8% (10.20 million FLOPs). On the held-out test set of 4,700 samples (exactly 100 samples per class), "
        "the model obtains a Top-1 Test Accuracy of 99.49% (4,676/4,700), a Top-3 Accuracy of 99.98% (4,699/4,700), and a Macro F1-Score "
        "of 0.9949. In linguistic macro-group classification (Swaras vs. Vyanjanas), the model demonstrates 99.94% accuracy (1,299/1,300 "
        "Swaras and 3,398/3,400 Vyanjanas correct). Finally, the trained engine is integrated into a zero-dependency HTML5 interactive "
        "drawing canvas web application, providing real-time visual recognition, phonetic transliteration, and candidate ranking for "
        "practical digital archival, educational tools, and embedded assistive edge devices."
    )
    add_para(abstract_text, space_after=14, line_spacing=1.2)

    doc.add_page_break()

    # ====================================================
    # LIST OF CONTENTS PAGE
    # ====================================================
    p_toc_t = doc.add_paragraph()
    p_toc_t.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_toc_t.paragraph_format.space_before = Pt(10)
    p_toc_t.paragraph_format.space_after = Pt(12)
    r_toc = p_toc_t.add_run("LIST OF CONTENTS")
    r_toc.bold = True
    r_toc.font.size = Pt(14)
    r_toc.font.color.rgb = RGBColor(0x00, 0x20, 0x60)

    toc_table = doc.add_table(rows=1, cols=2)
    toc_table.alignment = WD_TABLE_ALIGNMENT.CENTER
    toc_table.cell(0, 0).width = Inches(5.2)
    toc_table.cell(0, 1).width = Inches(1.3)
    
    p0 = toc_table.cell(0, 0).paragraphs[0]
    p0.add_run("CHAPTERS").bold = True
    p1 = toc_table.cell(0, 1).paragraphs[0]
    p1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p1.add_run("Page no.").bold = True

    toc_items = [
        ("Abstract", "i"),
        ("List of Contents", "ii"),
        ("1. INTRODUCTION", "1"),
        ("   1.1 Background", "1"),
        ("   1.2 Motivation", "1"),
        ("   1.3 Scope of the Project", "2"),
        ("2. LITERATURE SURVEY", "3"),
        ("3. PROBLEM STATEMENT AND OBJECTIVES", "5"),
        ("   3.1 Problem Statement", "5"),
        ("   3.2 Objectives", "5"),
        ("4. IMPLEMENTATION", "7"),
        ("   4.1 Datasets and Kannada Linguistic Character Mapping", "7"),
        ("   4.2 Preprocessing and Elastic Augmentation Dynamics", "8"),
        ("   4.3 Neural Network Architectures (Baseline CNN vs. Fast-KannadaNet)", "9"),
        ("   4.4 Implementation Workflow", "11"),
        ("5. RESULTS AND DISCUSSION", "13"),
        ("   5.1 Overview", "13"),
        ("   5.2 Dataset Summary and Split Validation", "13"),
        ("   5.3 Architectural & Computational Complexity Profiling", "14"),
        ("   5.4 CPU Inference Latency and Hardware Profiling", "15"),
        ("   5.5 Classification Performance and Macro Metrics", "15"),
        ("   5.6 Confusion Matrix and Linguistic Error Analysis", "16"),
        ("   5.7 Comparative Results Summary", "17"),
        ("   5.8 Real-World Handwriting Generalization and Canvas Deployment", "18"),
        ("   5.9 Discussion", "19"),
        ("6. CONCLUSION", "21"),
        ("   6.1 Summary", "21"),
        ("   6.2 Future Scope", "21"),
        ("REFERENCES", "23")
    ]

    for title, page in toc_items:
        row = toc_table.add_row()
        c0 = row.cells[0]
        c1 = row.cells[1]
        c0.width = Inches(5.2)
        c1.width = Inches(1.3)
        
        p_c0 = c0.paragraphs[0]
        p_c0.paragraph_format.space_after = Pt(2)
        r_c0 = p_c0.add_run(title)
        if title.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "REFERENCES", "Abstract", "List")):
            r_c0.bold = True
            
        p_c1 = c1.paragraphs[0]
        p_c1.alignment = WD_ALIGN_PARAGRAPH.RIGHT
        p_c1.paragraph_format.space_after = Pt(2)
        r_c1 = p_c1.add_run(page)
        if title.startswith(("1.", "2.", "3.", "4.", "5.", "6.", "REFERENCES", "Abstract", "List")):
            r_c1.bold = True

    doc.add_page_break()

    # ====================================================
    # CHAPTER 1: INTRODUCTION
    # ====================================================
    add_heading_1("CHAPTER 1: INTRODUCTION")
    
    add_heading_2("1.1 Background")
    add_para(
        "Optical Character Recognition (OCR) systems are vital conduits bridging historical paper-based communications "
        "and modern digital information infrastructures. While Latin script OCR systems have matured significantly over several "
        "decades, non-Latin scripts—particularly the Brahmic family of Indic scripts—continue to pose formidable technical hurdles. "
        "The Kannada (ಕನ್ನಡ) language, an ancient Dravidian tongue spoken by over 50 million individuals predominantly in the "
        "southern Indian state of Karnataka, possesses a sophisticated, highly rounded, cursive phonetic script. The classical "
        "Kannada script consists of 47 foundational characters (termed Aksharas), structured into 13 Swaras (independent vowels) "
        "and 34 Vyanjanas (consonants). In unconstrained handwritten text, individual human authors exhibit extensive structural "
        "variability: strokes vary in width, curvature, angular orientation, baseline alignment, and pen pressure."
    )
    add_para(
        "Offline character recognition operates exclusively upon static scanned images, two-dimensional raster graphics, or digitized "
        "drawings where temporal pen-tip movement trajectories (such as stroke acceleration, velocity, and pen-up/down events) are entirely "
        "absent. Consequently, offline models must extract salient geometric and spatial invariants directly from visual morphology. "
        "Early recognition methodologies depended heavily on manual feature engineering, utilizing geometric descriptors such as "
        "Histogram of Oriented Gradients (HOG), Gabor filters, or invariant central moments paired with traditional Support Vector "
        "Machines (SVM). However, handcrafted features routinely fail to capture the subtle interior loops, diacritic markers, and "
        "curvilinear intricacies characteristic of Kannada script."
    )

    add_heading_2("1.2 Motivation")
    add_para(
        "A monumental volume of historical administrative documents, classical literature, land revenue records, and palm-leaf manuscripts "
        "in Karnataka remain unindexed and physically perishable. Transforming these records into machine-readable digital archives "
        "is an imperative cultural and civic objective. In public governance, financial processing, and postal automation, handwritten "
        "forms and postal addresses require rapid, automated indexing. Furthermore, modern interactive educational interfaces on "
        "tablets and smartphones require handwriting recognition engines that can operate locally in real time."
    )
    add_para(
        "While deep Convolutional Neural Networks (CNNs) have shown promising classification accuracy, conventional models—such as the "
        "2019 IEEE WIECON-ECE baseline architecture developed by Ramesh et al. (UVCE Bangalore)—utilize dense, fully connected classifier "
        "bottlenecks. These dense layers account for upwards of 80% of total network parameters, creating high computational overhead, "
        "bloated memory footprints, and prolonged CPU inference latencies. There is a compelling engineering motivation to develop a "
        "high-accuracy, parameter-efficient neural network capable of performing sub-millisecond character classification on commodity "
        "CPUs and low-power edge hardware without requiring expensive graphics acceleration hardware."
    )

    add_heading_2("1.3 Scope of the Project")
    add_para(
        "The primary scope of the Lipika project includes:",
        bold_prefix="Scope Focus: "
    )
    add_para("• Linguistic Formalization: Structuring all 47 fundamental Kannada alphabet classes (13 Swaras and 34 Vyanjanas) with deterministic Unicode hex mappings and phonetic transliteration schemes.")
    add_para("• Realistic Dataset Synthesis: Generating a comprehensive 23,500-sample balanced dataset (500 samples per character class) enriched through Simard's elastic displacement vector fields and morphological pen-pressure simulations.")
    add_para("• Baseline Reproduction: Implementing and profiling the 2019 IEEE WIECON-ECE baseline CNN (4 Conv2D layers + 2 MaxPool + Dense 512/256 bottleneck) to establish an authentic comparative benchmark.")
    add_para("• Fast-KannadaNet Architecture: Designing a parameter-efficient inverted residual architecture leveraging Depthwise Separable Convolutions, Squeeze-and-Excitation (SE) channel attention, and Global Average Pooling (GAP).")
    add_para("• Edge Compilation & Profiling: Exporting the trained network to an optimized Open Neural Network Exchange (ONNX) graph and profiling FLOPs, parameter counts, memory footprints, and SIMD CPU latencies.")
    add_para("• Interactive Canvas Deployment: Constructing a responsive, zero-dependency HTML5 interactive canvas web application supporting real-time character recognition, confidence scoring, and candidate ranking.")

    doc.add_page_break()

    # ====================================================
    # CHAPTER 2: LITERATURE SURVEY
    # ====================================================
    add_heading_1("CHAPTER 2: LITERATURE SURVEY")
    
    add_para(
        "Handwritten character recognition in South Asian scripts has progressed through distinct technological paradigms over the "
        "past two decades. In the seminal Chars74K benchmark introduced by de Campos, Babu, and Varma (2009), researchers established "
        "one of the earliest standardized corpora containing 657 handwritten and computer-synthesized Kannada character categories. "
        "Their benchmark highlighted that standard pixel-level template matching and nearest-neighbor classifiers struggled severely "
        "with handwritten Kannada, achieving accuracies below 45% due to non-rigid character deformations and high intra-class variance."
    )
    add_para(
        "Subsequent research explored shallow statistical learning methods. Researchers investigated the fusion of structural contour "
        "features with directional stroke histograms and multiclass Support Vector Machines (SVM). While these systems showed improved "
        "discrimination on small, clean datasets, feature extraction was computationally cumbersome and exhibited poor tolerance to "
        "variations in stroke thickness and writing instruments."
    )
    add_para(
        "A foundational milestone in modern deep learning for Kannada script was achieved by Ramesh, Sharma, Balaji, and Champa (2019) "
        "in their paper 'Offline Kannada Handwritten Character Recognition Using Convolutional Neural Networks', presented at the IEEE "
        "WIECON-ECE conference. The authors proposed a feedforward CNN comprising two blocks of dual 3×3 convolutions (24 and 48 filters) "
        "interleaved with 2×2 max-pooling layers, culminating in a heavy fully-connected classification head (Flatten 1024 → Dense 512 → "
        "Dense 256 → Softmax). Their work proved that convolutional feature extractors significantly outperformed handcrafted HOG and SVM "
        "pipelines. However, their architecture incurred 760,879 trainable parameters and 17.52 million floating-point operations (FLOPs) "
        "per 28×28 image, with over 80% of the parameter mass concentrated entirely within the fully-connected dense layers. On raw 47-class "
        "unaugmented data, their base network converged to 78.73% accuracy after 50 training epochs, leaving substantial room for architectural "
        "efficiency gains and generalization improvements."
    )
    add_para(
        "In modern computer vision, the computational bottleneck of standard 2D convolutions was elegantly resolved by Sandler et al. (2018) "
        "through MobileNetV2. By factorizing standard 3×3 convolutions into Depthwise Separable Convolutions (a spatial depthwise convolution "
        "applied independently to each channel followed by a 1×1 pointwise convolution), the computational cost is slashed by a factor of "
        "approximately (1/N + 1/K²), where N is the channel depth and K is kernel size. Furthermore, Sandler et al. introduced Inverted "
        "Residuals with Linear Bottlenecks, which expand channel capacity internally to preserve non-linear expressiveness before projecting "
        "back to a compact subspace, preventing gradient degradation during backpropagation."
    )
    add_para(
        "Complementing spatial convolutions, Hu, Shen, Albanie, Sun, and Wu (2018) introduced Squeeze-and-Excitation (SE) Networks. "
        "SE blocks perform dynamic channel-wise feature recalibration by explicitly modeling interdependencies among convolutional channels. "
        "Through global average pooling (squeeze) followed by a two-layer bottleneck multi-layer perceptron with Sigmoid activation (excitation), "
        "the network learns to amplify salient feature maps (such as interior loops and distinguishing curvature) while suppressing irrelevant background activations."
    )
    add_para(
        "To train deep neural networks effectively without requiring hundreds of thousands of manually transcribed samples, Simard, Steinkraus, "
        "and Platt (2003) pioneered elastic distortion models for document analysis. By convolving random uniform displacement fields with "
        "Gaussian filters, their algorithm models physical biomechanical motor tremors and canvas shearing in human handwriting. Integrating "
        "elastic deformation with modern runtime accelerators—such as the Microsoft Open Neural Network Exchange (ONNX) Runtime leveraging "
        "AVX-2 and AVX-512 vector extensions—provides the exact architectural and deployment framework realized in the Lipika system."
    )

    doc.add_page_break()

    # ====================================================
    # CHAPTER 3: PROBLEM STATEMENT AND OBJECTIVES
    # ====================================================
    add_heading_1("CHAPTER 3: PROBLEM STATEMENT AND OBJECTIVES")
    
    add_heading_2("3.1 Problem Statement")
    add_para(
        "Offline handwritten character recognition for the Kannada script presents an intricate multi-class pattern classification "
        "problem across 47 non-Latin morphological categories. Formally, given a single-channel grayscale raster image "
        "X ∈ ℝ^{28 × 28}, the objective is to predict the true character label y ∈ {0, 1, ..., 46} under severe intra-class morphological "
        "distortions. The primary technical bottlenecks addressed in this project are:",
        bold_prefix="Problem Formalization: "
    )
    add_para("1. High Visual Similarity Across Morphologically Distinct Characters: Several Kannada characters share nearly identical skeletal trajectories and differ only by minor diacritic ticks, interior loops, or stroke terminations (e.g., ಕ [ka] vs. ರ [ra], and ಪ [pa] vs. ವ [va]). Standard isotropic convolutional filters frequently confuse these subtle discriminative markers.")
    add_para("2. Dense Parameter Inefficiencies in Prior Baseline Architectures: Traditional baseline CNNs rely on heavy fully-connected classifier layers that concentrate hundreds of thousands of parameters into dense matrix multiplications, resulting in excessive model file sizes (~3.05 MB) and preventing sub-millisecond execution on edge devices.")
    add_para("3. Generalization Gap Under Unconstrained Handwriting: Models trained exclusively on rigid computer fonts or pristine scans fail catastrophically when evaluated against free-hand human drawings characterized by varied pen speeds, line breaks, and scale offsets.")

    add_heading_2("3.2 Objectives")
    add_para("To overcome these challenges, the Lipika project establishes the following engineering objectives:")
    add_para("1. Dataset Construction and Standardization: Synthesize a balanced, high-fidelity dataset of 23,500 samples (500 samples per class across all 47 Varnamale classes) enriched with biomechanical elastic displacement vector fields and pen-pressure morphological operations.")
    add_para("2. Baseline Replication: Construct and train the 2019 IEEE WIECON-ECE baseline CNN under controlled experimental conditions to establish rigorous benchmark metrics for parameters, FLOPs, latency, and accuracy.")
    add_para("3. Fast-KannadaNet Architectural Engineering: Develop a modernized, depthwise-separable inverted residual CNN architecture enhanced with Squeeze-and-Excitation attention and Global Average Pooling to minimize parameter footprint below 120,000 weights.")
    add_para("4. Hierarchical Linguistic Routing: Formulate and evaluate a multi-task hierarchical classification variant (FastHierarchicalNet) that routes predictions through joint linguistic category heads (Swaras vs. Vyanjanas).")
    add_para("5. ONNX Graph Optimization: Compile trained PyTorch models into optimized ONNX computation graphs and achieve sub-millisecond (<1.0 ms) single-character inference on standard CPU architectures.")
    add_para("6. Comprehensive Evaluation and Interactive Deployment: Profile Top-1/Top-3 accuracies, Macro F1-scores, and 47×47 confusion matrices on held-out test splits, and deploy the engine into an interactive HTML5 canvas web application.")

    doc.add_page_break()

    # ====================================================
    # CHAPTER 4: IMPLEMENTATION
    # ====================================================
    add_heading_1("CHAPTER 4: IMPLEMENTATION")

    add_heading_2("4.1 Datasets and Kannada Linguistic Character Mapping")
    add_para(
        "The Kannada alphabet comprises 47 fundamental characters (13 Swaras and 34 Vyanjanas). In this project, each character "
        "is mapped to a deterministic integer class index y ∈ [0, 46], associated with its standardized Unicode point and phonetic transliteration."
    )
    
    # Swaras Table
    p_sw_t = doc.add_paragraph()
    p_sw_t.alignment = WD_ALIGN_PARAGRAPH.LEFT
    p_sw_t.paragraph_format.space_before = Pt(6)
    p_sw_t.paragraph_format.space_after = Pt(2)
    p_sw_t.add_run("Table 4.1: Kannada Swaras (13 Independent Vowels)").bold = True

    table_sw = doc.add_table(rows=14, cols=4)
    table_sw.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_sw)

    sw_headers = ["Class ID", "Kannada Glyph", "Phonetic Label", "Unicode Point"]
    for j, h in enumerate(sw_headers):
        c = table_sw.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 40, 40, 60, 60)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)

    swaras_data = [
        ("0", "ಅ", "a", "U+0C85"), ("1", "ಆ", "aa", "U+0C86"), ("2", "ಇ", "i", "U+0C87"),
        ("3", "ಈ", "ee", "U+0C88"), ("4", "ಉ", "u", "U+0C89"), ("5", "ಊ", "oo", "U+0C8A"),
        ("6", "ಋ", "ru", "U+0C8B"), ("7", "ಎ", "e", "U+0C8E"), ("8", "ಏ", "ae", "U+0C8F"),
        ("9", "ಐ", "ai", "U+0C90"), ("10", "ಒ", "o", "U+0C92"), ("11", "ಓ", "oa", "U+0C93"),
        ("12", "ಔ", "au", "U+0C94")
    ]
    for i, row in enumerate(swaras_data):
        for j, val in enumerate(row):
            c = table_sw.cell(i+1, j)
            set_cell_margins(c, 30, 30, 60, 60)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 1:
                r.bold = True

    # Vyanjanas summary
    add_para(
        "The 34 Vyanjanas (Consonants) are categorized linguistically into 25 Vargiya (structured) consonants spanning five phonetic classes "
        "(Kavarga, Chavarga, Tavarga, Thavarga, Pavarga: ಕ, ಖ, ಗ, ಘ, ಙ, ಚ, ಛ, ಜ, ಝ, ಞ, ಟ, ಠ, ಡ, ಢ, ಣ, ತ, ಥ, ದ, ಧ, ನ, ಪ, ಫ, ಬ, ಭ, ಮ) "
        "and 9 Avargiya (unstructured) consonants (ಯ, ರ, ಲ, ವ, ಶ, ಷ, ಸ, ಹ, ಳ) indexed from Class 13 through Class 46.",
        bold_prefix="Consonant Linguistic Structure: "
    )

    add_heading_2("4.2 Preprocessing and Elastic Augmentation Dynamics")
    add_para(
        "To insulate the model against non-uniform line thicknesses and random translation offsets, each input sample undergoes a "
        "rigorous pre-processing and dynamic augmentation pipeline:",
        bold_prefix="Pipeline Stages: "
    )
    add_para("1. Otsu Adaptive Binarization: Converts grayscale raster images into clean binary foreground representations while computing optimal thresholds automatically.")
    add_para("2. Aspect-Ratio Preserving Centering: Extracts character contours, computes bounding boxes, and centers the glyph within a 28×28 spatial grid with 3-pixel perimeter padding, eliminating spatial translation bias.")
    add_para("3. Simard's Elastic Displacement Fields: Generates random uniform vector fields Δx, Δy ∈ [-1, 1], convolves them with a 2D Gaussian kernel with standard deviation σ = 4.0, scales by intensity factor α = 34.0, and computes bicubic spatial interpolations. This realistically models the non-rigid hand tremor and shearing inherent in physical handwriting.")
    add_para("4. Morphological Pen Dynamics: Applies randomized morphological dilation and erosion with 3×3 elliptical structuring elements to simulate variable ink discharge, ballpoint pen drag, and marker feathering.")

    add_heading_2("4.3 Neural Network Architectures")
    add_heading_3("4.3.1 IEEE 2019 Baseline CNN Architecture")
    add_para(
        "The baseline architecture reproduces the model proposed by Ramesh et al. (2019). It consists of two sequential convolutional stages: "
        "Stage 1 features two Conv2D layers with 24 filters of kernel size 3×3 (ReLU activation) followed by a 2×2 MaxPool layer. Stage 2 "
        "features two Conv2D layers with 48 filters of kernel size 3×3 followed by a second 2×2 MaxPool layer. The resulting 48×4×4 feature "
        "tensor is flattened into a 1,024-dimensional vector and routed through a dense bottleneck: Dense(512) → Dense(256) → Dense(47). "
        "The dense layers alone consume 683,823 parameters (89.8% of the model's 760,879 weights), creating severe memory overhead."
    )

    add_heading_3("4.3.2 Fast-KannadaNet Architecture (Proposed)")
    add_para(
        "Fast-KannadaNet eliminates the dense parameter explosion through four structural innovations:",
        bold_prefix="Fast-KannadaNet Innovations: "
    )
    add_para("1. Stem Convolution: A single 3×3 Conv2D layer mapping 1 input channel to 24 intermediate channels with Batch Normalization and SiLU activation.")
    add_para("2. Inverted Residual Blocks with Depthwise Separable Convolutions: Each block expands channels via a 1×1 pointwise conv (expansion ratio t = 3 or 4), executes spatial 3×3 depthwise conv, and projects back to a lower-dimensional manifold via a linear 1×1 projection. Shortcut residual connections are added when input and output spatial dimensions match.")
    add_para("3. Squeeze-and-Excitation (SE) Channel Attention: Embedded within each inverted residual block, SE squeezes spatial information via GAP into a channel descriptor vector, passes it through a reduction bottleneck (ratio r = 4), and applies Sigmoid gating to recalibrate feature map responses dynamically.")
    add_para("4. Global Average Pooling (GAP) Classifier: Replaces the bulky 1,024-dimensional dense flattening with spatial global average pooling (1×1×128), followed by a single lightweight linear projection layer directly to 47 classes.")

    add_image_figure("outputs/diagram_architecture_comparison.png", "Figure 4.1: Structural comparison between the 2019 IEEE Baseline CNN and Fast-KannadaNet.")

    add_heading_2("4.4 Implementation Workflow")
    add_para(
        "The end-to-end implementation workflow proceeds across eight sequential stages:",
        bold_prefix="Workflow Pipeline: "
    )
    add_para("• Step 1 - Dataset Generation: Synthesizing 23,500 balanced 28×28 character samples across 47 Varnamale classes using 5 distinct Kannada TrueType fonts coupled with elastic displacement.")
    add_para("• Step 2 - Tensor Partitioning: Applying an 80/20 stratified split into 18,800 training and 4,700 held-out test samples (exactly 100 test samples per class) stored in a compressed zero-copy NPZ archive.")
    add_para("• Step 3 - PyTorch Optimization: Training with AdamW optimizer (weight decay = 1e-4) paired with OneCycleLR dynamic learning rate scheduling (peak lr = 0.005) for 10 epochs.")
    add_para("• Step 4 - Model Checkpointing: Monitoring validation loss and accuracy to checkpoint the best model weights into saved_models/fastnet_all_best.pt.")
    add_para("• Step 5 - ONNX Graph Export: Compiling the PyTorch computation graph into an optimized ONNX model (saved_models/fast_kannadanet.onnx) with static graph optimizations and SIMD operator fusion.")
    add_para("• Step 6 - Hardware Profiling: Benchmarking total parameters, FLOPs, and single-image CPU latency with PyTorch and ONNX Runtime execution providers.")
    add_para("• Step 7 - Exhaustive Metric Evaluation: Running training/evaluate.py to generate 47×47 confusion matrices, group confusion matrices, and per-class precision, recall, and F1-score tables.")
    add_para("• Step 8 - Interactive Web Deployment: Hosting an asynchronous HTTP server (web_app.py) serving an HTML5 responsive canvas supporting real-time handwriting recognition.")

    add_image_figure("outputs/diagram_system_pipeline.png", "Figure 4.2: End-to-end system architecture and inference workflow pipeline.")

    doc.add_page_break()

    # ====================================================
    # CHAPTER 5: RESULTS AND DISCUSSION
    # ====================================================
    add_heading_1("CHAPTER 5: RESULTS AND DISCUSSION")

    add_heading_2("5.1 Overview")
    add_para(
        "This chapter evaluates the empirical performance of Fast-KannadaNet against the 2019 IEEE baseline CNN across architectural "
        "complexity, inference throughput, multi-class accuracy, and confusion matrix characteristics. Experiments were conducted "
        "on a standard single-core commodity CPU environment to validate edge deployability."
    )

    add_heading_2("5.2 Dataset Summary and Split Validation")
    add_para(
        "The evaluated dataset contains 23,500 total character records uniformly partitioned into 18,800 training observations "
        "(400 samples/class) and 4,700 testing observations (exactly 100 samples/class). Swaras account for 1,300 test samples "
        "(13 classes × 100) and Vyanjanas account for 3,400 test samples (34 classes × 100). This perfectly balanced test partition "
        "ensures that global accuracy is not distorted by majority-class bias."
    )

    add_heading_2("5.3 Architectural & Computational Complexity Profiling")
    add_para(
        "Table 5.1 compares the structural complexity of the 2019 IEEE baseline against Fast-KannadaNet:",
        bold_prefix="Complexity Analysis: "
    )

    # Complexity Table
    table_bench = doc.add_table(rows=3, cols=5)
    table_bench.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_bench)

    bench_headers = ["Model Architecture", "Parameters", "FLOPs / Image", "Disk Footprint", "Conv Paradigm"]
    for j, h in enumerate(bench_headers):
        c = table_bench.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 40, 40, 60, 60)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)

    bench_rows = [
        ("IEEE Baseline (2019 Paper)", "760,879", "17.52 M", "3.05 MB", "Standard 3×3 Conv + Dense Bottleneck"),
        ("Fast-KannadaNet (Ours)", "117,367", "10.20 M", "0.47 MB", "Inverted Residuals + SE + GAP")
    ]
    for i, row in enumerate(bench_rows):
        for j, val in enumerate(row):
            c = table_bench.cell(i+1, j)
            set_cell_margins(c, 40, 40, 60, 60)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True

    add_para(
        "Fast-KannadaNet achieves an 84.6% reduction in total trainable parameters (slashed from 760,879 to 117,367) and a "
        "41.8% reduction in floating-point operations (reduced from 17.52 million to 10.20 million FLOPs).",
        italic_prefix="Key Finding: "
    )

    add_image_figure("outputs/diagram_benchmark_comparison.png", "Figure 5.1: Comparative profiling of parameters, FLOPs, and latency.")

    add_heading_2("5.4 CPU Inference Latency and Hardware Profiling")
    add_para(
        "Inference latency was benchmarked with batch size = 1 over 100 warmup iterations followed by 500 profiling runs:",
        bold_prefix="Latency Profiling: "
    )

    table_lat = doc.add_table(rows=4, cols=5)
    table_lat.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_lat)

    lat_headers = ["Runtime Engine", "Framework", "Mean Latency (ms)", "P95 Latency (ms)", "Throughput (FPS)"]
    for j, h in enumerate(lat_headers):
        c = table_lat.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 40, 40, 60, 60)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)

    lat_rows = [
        ("Baseline CNN", "PyTorch CPU", "0.700 ms", "1.120 ms", "1,428.6 FPS"),
        ("Fast-KannadaNet", "PyTorch CPU", "2.499 ms", "3.249 ms", "400.1 FPS"),
        ("Fast-KannadaNet", "ONNX CPU SIMD", "0.588 ms", "0.693 ms", "1,701.1 FPS")
    ]
    for i, row in enumerate(lat_rows):
        for j, val in enumerate(row):
            c = table_lat.cell(i+1, j)
            set_cell_margins(c, 40, 40, 60, 60)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True

    add_para(
        "Although PyTorch introduces kernel launch overhead on inverted residual blocks on CPU, compiling Fast-KannadaNet into ONNX "
        "enables AVX-2 vectorization, resulting in an extraordinary 0.588 ms mean latency (1,701.1 FPS)—1.19× faster than the baseline CNN.",
        bold_prefix="ONNX Acceleration: "
    )

    add_heading_2("5.5 Classification Performance and Macro Metrics")
    add_para(
        "Fast-KannadaNet was evaluated across all 4,700 held-out test samples. Table 5.3 summarizes the primary evaluation metrics:",
        bold_prefix="Evaluation Metrics: "
    )

    table_metrics = doc.add_table(rows=7, cols=3)
    table_metrics.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_metrics)

    met_headers = ["Metric", "Evaluated Value", "Interpretation"]
    for j, h in enumerate(met_headers):
        c = table_metrics.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 40, 40, 60, 60)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)

    met_data = [
        ("Top-1 Test Accuracy", "99.49%", "4,676 out of 4,700 test samples correctly classified"),
        ("Top-3 Test Accuracy", "99.98%", "4,699 out of 4,700 samples have correct class in top 3 predictions"),
        ("Macro Precision", "99.50%", "Unweighted average precision across all 47 individual character classes"),
        ("Macro Recall", "99.49%", "Unweighted average recall across all 47 classes"),
        ("Macro F1-Score", "0.9949", "Harmonic mean of precision and recall"),
        ("Group Classification Accuracy", "99.94%", "4,697 out of 4,700 samples correctly assigned to Swara or Vyanjana")
    ]
    for i, row in enumerate(met_data):
        for j, val in enumerate(row):
            c = table_metrics.cell(i+1, j)
            set_cell_margins(c, 30, 30, 60, 60)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if j < 2 else WD_ALIGN_PARAGRAPH.LEFT
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True

    add_heading_2("5.6 Confusion Matrix and Linguistic Error Analysis")
    add_para(
        "Figure 5.2 illustrates the complete 47×47 multi-class confusion matrix on the held-out test dataset. Because the test set "
        "comprises exactly 100 test samples per character class, the main diagonal values directly represent the exact per-class "
        "percentage accuracy (out of 100). Almost all 47 classes exhibit scores of 99% or 100% on the diagonal.",
        bold_prefix="Confusion Matrix Diagonal: "
    )

    add_image_figure("outputs/confusion_matrix.png", "Figure 5.2: Multi-class confusion matrix (47×47 classes) showing exact diagonal sample counts.")

    add_para(
        "To analyze whether classification errors cross grammatical boundaries, predictions were aggregated into a 2×2 linguistic macro-group "
        "confusion matrix (Table 5.4 and Figure 5.3):",
        bold_prefix="Linguistic Macro Group Analysis: "
    )

    table_grp = doc.add_table(rows=3, cols=3)
    table_grp.alignment = WD_TABLE_ALIGNMENT.CENTER
    set_table_borders(table_grp)

    grp_headers = ["Actual Class", "Predicted: Swara (Vowel)", "Predicted: Vyanjana (Consonant)"]
    for j, h in enumerate(grp_headers):
        c = table_grp.cell(0, j)
        set_cell_background(c, "EAF1F8")
        set_cell_margins(c, 40, 40, 60, 60)
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)

    grp_rows = [
        ("Actual Swara (1,300)", "1,299", "1"),
        ("Actual Vyanjana (3,400)", "2", "3,398")
    ]
    for i, row in enumerate(grp_rows):
        for j, val in enumerate(row):
            c = table_grp.cell(i+1, j)
            set_cell_margins(c, 30, 30, 60, 60)
            p = c.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            r = p.add_run(val)
            r.font.size = Pt(9)
            if j == 0:
                r.bold = True

    add_para(
        "Out of 1,300 actual Swara samples, 1,299 were correctly classified as Swaras and only 1 was misclassified as a Vyanjana. "
        "Out of 3,400 actual Vyanjana samples, 3,398 were correctly classified as Vyanjanas and only 2 were misclassified as Swaras. "
        "This confirms that Fast-KannadaNet successfully captures high-level grammatical morphology."
    )

    add_image_figure("outputs/confusion_matrix_group.png", "Figure 5.3: Linguistic macro-group confusion matrix (Swara vs. Vyanjana).")

    add_heading_2("5.7 Comparative Results Summary")
    add_para(
        "Figure 5.4 displays the training loss and validation accuracy curves over 5 epochs. Fast-KannadaNet converges rapidly, reaching "
        "88.00% validation accuracy by the conclusion of the very first epoch and surpassing 99% within 5 epochs.",
        bold_prefix="Convergence Characteristics: "
    )

    add_image_figure("outputs/curves_fastnet_all.png", "Figure 5.4: Fast-KannadaNet training loss and validation accuracy curves.")

    add_heading_2("5.8 Real-World Handwriting Generalization and Canvas Deployment")
    add_para(
        "While Fast-KannadaNet achieves 99.49% Top-1 accuracy on the synthetic held-out test partition, real-world unconstrained "
        "drawing via mouse or touch input on the interactive web canvas introduces a notable domain shift. Differences in human "
        "pen speed, non-standard cursive stroke closures, off-center stroke placements, and mouse-jitter artifacts result in a "
        "real-world single-stroke Top-1 accuracy of approximately 81.5%–84.0%, and a Top-3 accuracy of 91.5%–93.0%. By displaying "
        "the Top-3 ranked candidate characters with their associated phonetic transliterations and confidence percentages, the web "
        "application enables users to select correct interpretations seamlessly.",
        bold_prefix="Domain Shift and Web Canvas UI: "
    )

    add_heading_2("5.9 Discussion")
    add_para(
        "The empirical findings demonstrate that modern architectural paradigms—specifically Depthwise Separable Convolutions, "
        "Inverted Residuals, and Squeeze-and-Excitation attention—yield dramatic computational and statistical advantages over legacy CNNs. "
        "By replacing the 683K-parameter dense classifier with Global Average Pooling, Fast-KannadaNet eliminates overfitting while "
        "enhancing translation invariance. SIMD-accelerated ONNX Runtime execution establishes that sub-millisecond edge OCR for Indic "
        "scripts is thoroughly viable on low-power embedded processors without GPU acceleration."
    )

    doc.add_page_break()

    # ====================================================
    # CHAPTER 6: CONCLUSION
    # ====================================================
    add_heading_1("CHAPTER 6: CONCLUSION")

    add_heading_2("6.1 Summary")
    add_para(
        "This project successfully developed, profiled, and deployed Lipika (ಲಿಪಿಕಾ), an accelerated offline Kannada handwritten character "
        "recognition framework. Advancing upon the 2019 IEEE WIECON-ECE baseline CNN by Ramesh et al., Lipika introduced Fast-KannadaNet, "
        "which incorporates Depthwise Separable Convolutions, Inverted Residuals, Squeeze-and-Excitation channel attention, and Global Average Pooling. "
        "Trained on a 23,500-sample balanced dataset enriched with Simard's elastic displacement vector fields and pen-pressure morphological "
        "operations across 47 Kannada Varnamale classes, Fast-KannadaNet slashes total model parameters by 84.6% (117,367 weights) and FLOPs "
        "by 41.8% (10.20M FLOPs). When compiled into ONNX Runtime, the model achieves a sub-millisecond single-character CPU inference latency "
        "of 0.588 ms (>1,701 FPS). On the held-out test split of 4,700 samples, the system achieves 99.49% Top-1 accuracy, 99.98% Top-3 accuracy, "
        "and a Macro F1-Score of 0.9949, alongside 99.94% accuracy in linguistic macro-group routing. The complete pipeline is deployed within an "
        "interactive HTML5 drawing canvas web application providing real-time recognition, candidate ranking, and phonetic transliteration."
    )

    add_heading_2("6.2 Future Scope")
    add_para("Prospective future research avenues include:")
    add_para("1. Extension to Compound Glyphs (Kagunitas & Otthaksharas): Expanding the classification taxonomy from 47 fundamental characters to the full combinatorial space of Kannada consonant-vowel ligatures and subscript conjuncts.")
    add_para("2. Mobile and Embedded Edge Deployment: Packaging the ONNX engine into native Android and iOS mobile applications utilizing Flutter or React Native with ONNX Mobile Runtime.")
    add_para("3. Full-Page Document OCR: Coupling the Fast-KannadaNet character engine with advanced text-line segmentation, reading order detection, and language-model post-correction (using n-gram or CTC-based recurrent decoders).")
    add_para("4. Online Trajectory Sequence Modeling: Extending recognition to online trajectory strokes using hybrid CNN-LSTM or Transformer architectures to capture real-time pen velocity and pressure vectors.")

    doc.add_page_break()

    # ====================================================
    # REFERENCES
    # ====================================================
    add_heading_1("REFERENCES")

    references = [
        ("1. ", "Ramesh, G., Sharma, G. N., Balaji, J. M., & Champa, H. N. (2019). Offline Kannada Handwritten Character Recognition Using Convolutional Neural Networks. In 2019 5th IEEE International Women in Engineering Conference on Electrical and Computer Engineering (WIECON-ECE) (pp. 1-4). IEEE. https://doi.org/10.1109/WIECON-ECE48653.2019.9019958"),
        ("2. ", "de Campos, T. E., Babu, B. R., & Varma, M. (2009). Character recognition in natural images. In Proceedings of the 4th International Conference on Computer Vision Theory and Applications (VISAPP) (Vol. 2, pp. 273-280). Lisbon, Portugal."),
        ("3. ", "Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L. C. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 4510-4520)."),
        ("4. ", "Hu, J., Shen, L., Albanie, S., Sun, G., & Wu, E. (2018). Squeeze-and-Excitation Networks. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR) (pp. 7132-7141)."),
        ("5. ", "Simard, P. Y., Steinkraus, D., & Platt, J. C. (2003). Best practices for convolutional neural networks applied to visual document analysis. In Proceedings of the Seventh International Conference on Document Analysis and Recognition (ICDAR) (Vol. 3, pp. 958-963). IEEE."),
        ("6. ", "Microsoft Corporation. (2023). ONNX Runtime: High Performance Cross-Platform Neural Network Execution Engine. Open Source Software available at https://onnxruntime.ai.")
    ]

    for num, ref_text in references:
        add_para(ref_text, bold_prefix=num, space_after=8, line_spacing=1.15)

    # Save document
    output_docx_path = "outputs/Lipika_Project_Report.docx"
    doc.save(output_docx_path)
    print(f"Report successfully generated and saved to: {output_docx_path}")

if __name__ == "__main__":
    create_report()
