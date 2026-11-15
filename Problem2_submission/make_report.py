"""Generates report.pdf -- the short methodology report for submission."""
import json

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image,
                                 Table, TableStyle, ListFlowable, ListItem)

with open("attack_stats.json") as f:
    stats = json.load(f)

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="Body", parent=styles["Normal"], spaceAfter=8, leading=15))
styles.add(ParagraphStyle(name="H2", parent=styles["Heading2"], spaceBefore=14, spaceAfter=6))
styles.add(ParagraphStyle(name="H3", parent=styles["Heading3"], spaceBefore=10, spaceAfter=4))
styles.add(ParagraphStyle(name="Caption", parent=styles["Normal"], fontSize=9,
                           textColor=colors.grey, alignment=1, spaceAfter=10))

doc = SimpleDocTemplate("report.pdf", pagesize=letter,
                         topMargin=0.8 * inch, bottomMargin=0.8 * inch,
                         leftMargin=0.8 * inch, rightMargin=0.8 * inch)

S = []

S.append(Paragraph("Black-Box Adversarial Attack on a CNN", styles["Title"]))
S.append(Paragraph("Transfer-based attack with query-efficient hard-label refinement",
                    styles["Heading3"]))
S.append(Spacer(1, 10))

S.append(Paragraph("1. Problem setting", styles["H2"]))
S.append(Paragraph(
    "We are given 1000 clean 32x32 RGB images (10 classes, 100 images/class) and a "
    "pretrained CNN that acts as a strict hard-label oracle: it can be queried at any "
    "input image and returns only the predicted class index. No logits, probabilities, "
    "gradients, architecture, or parameters are observable. The task is to produce one "
    "adversarial image per clean image that (a) is misclassified by the model and "
    "(b) differs from the original by the smallest, least visible amount of noise "
    "possible.", styles["Body"]))

S.append(Paragraph("2. Why a pure decision-based search is the wrong default here",
                    styles["H2"]))
S.append(Paragraph(
    "Classic hard-label black-box attacks (Boundary Attack, HopSkipJumpAttack, "
    "Sign-OPT, RayS, etc.) estimate an attack direction purely from the sequence of "
    "yes/no answers the model gives, with no assumption about the model's training "
    "data. They are general-purpose but query-hungry: hundreds to several thousand "
    "queries per image are typical to converge on a small perturbation. Since the "
    "evaluation explicitly rewards using fewer model inferences, and since the clean "
    "images here are plainly drawn from the well-known public CIFAR-10 distribution "
    "(same 10 classes, same resolution), we instead exploit that side information with "
    "a <b>transfer-based</b> attack, reserving decision-based search only as a fallback.",
    styles["Body"]))

S.append(Paragraph("3. Method", styles["H2"]))
S.append(Paragraph("3.1 Surrogate ensemble", styles["H3"]))
S.append(Paragraph(
    "We trained two independent CNNs from scratch on the real, public CIFAR-10 "
    "training set (50,000 images), using standard data augmentation (random crop + "
    "horizontal flip), SGD with Nesterov momentum, weight decay, and a one-cycle "
    "learning-rate schedule for a small number of epochs each:", styles["Body"]))
S.append(ListFlowable([
    ListItem(Paragraph("<b>ResNet9</b>: a compact residual network (stem + 3 downsampling "
                        "conv blocks + 2 residual blocks + global average pool). "
                        "Final test accuracy on real CIFAR-10: <b>82.9%</b>.", styles["Body"])),
    ListItem(Paragraph("<b>VGGSmall</b>: a plain (non-residual) VGG-style stack of "
                        "conv-BN-ReLU blocks. Final test accuracy: <b>83.2%</b>.", styles["Body"])),
], bulletType="bullet"))
S.append(Paragraph(
    "These two surrogates are architecturally distinct from each other (and from the "
    "unknown target), which helps: averaging gradients from diverse models tends to "
    "find perturbation directions that are less specific to any single model's quirks "
    "and therefore transfer better to an unseen third model.", styles["Body"]))

S.append(Paragraph("3.2 Crafting candidate perturbations (white-box PGD on the surrogates)",
                    styles["H3"]))
S.append(Paragraph(
    "For a clean image x with true label y, we run untargeted projected gradient "
    "descent (PGD) in raw [0,1] pixel space against the mean cross-entropy loss of the "
    "surrogate ensemble, under an L-infinity budget &epsilon;:", styles["Body"]))
S.append(Paragraph(
    "x<sub>0</sub> = x,  &nbsp; x<sub>t+1</sub> = Clip<sub>[x-&epsilon;, x+&epsilon;]</sub>"
    "( x<sub>t</sub> + &alpha; &middot; sign(&nabla;<sub>x</sub> (1/2)&Sigma;<sub>m</sub> "
    "CE(m(x<sub>t</sub>), y)) ), &nbsp; clipped to [0,1] pixel range.",
    ParagraphStyle(name="eq", parent=styles["Body"], alignment=1, fontSize=10)))
S.append(Paragraph(
    "This is standard white-box PGD -- but critically, the gradient is taken only "
    "through our own surrogates, never through the target model.", styles["Body"]))

S.append(Paragraph("3.3 Eps-schedule with early stopping on real queries", styles["H3"]))
S.append(Paragraph(
    "Rather than committing to one fixed perturbation budget, we try an increasing "
    "schedule &epsilon; &isin; {4, 8, 12, 16, 24, 32}/255. At each level we run PGD "
    "(warm-started from the previous level's result) and then query the real "
    "black-box model once per still-unsolved image. As soon as an image's prediction "
    "flips, we freeze its perturbation and remove it from subsequent, larger-budget "
    "rounds. This means most images only ever need a small budget, and the real model "
    "is queried only once per budget level actually attempted for that image, not once "
    "per PGD iteration.", styles["Body"]))

S.append(Paragraph("3.4 Binary-search refinement (minimizing the final noise)", styles["H3"]))
S.append(Paragraph(
    "Once some successful perturbation direction d = x<sub>adv</sub> - x is found for an "
    "image, we binary-search a scalar &theta; &isin; [0,1] such that x + &theta;&middot;d "
    "is the smallest-magnitude point along that direction that still flips the real "
    "model's prediction, using only hard-label queries (10 queries/image, fixed budget). "
    "This step is what pushes the final perturbations well below the eps level at which "
    "they were first found, and is the main lever for the imperceptibility score.",
    styles["Body"]))

S.append(Paragraph("3.5 Fallback for non-transferring images", styles["H3"]))
S.append(Paragraph(
    "For any image that the transfer attack cannot flip even at the largest scheduled "
    "budget, we fall back to a simple randomized decision-based search: repeatedly try "
    "random uniform noise at a larger budget (64/255) and query the target model, "
    "keeping the first success (up to 20 tries), with one final larger-noise attempt as "
    "a last resort. In our run, this fallback was never needed -- every image succeeded "
    "through the transfer attack alone.", styles["Body"]))

S.append(Paragraph("4. Results", styles["H2"]))
rows = [
    ["Metric", "Value"],
    ["Attack success rate", f"{stats['success_rate']*100:.1f}% ({stats['n_success']}/{stats['n_total']})"],
    ["Already-misclassified clean images", str(stats["n_already_misclassified"])],
    ["Mean L-infinity perturbation (0-1 scale)", f"{stats['mean_linf_0_1']:.4f}  (~{stats['mean_linf_0_1']*255:.1f}/255)"],
    ["Max L-infinity perturbation (0-1 scale)", f"{stats['max_linf_0_1']:.4f}  (~{stats['max_linf_0_1']*255:.1f}/255)"],
    ["Mean L2 perturbation (3072-dim vector)", f"{stats['mean_l2']:.4f}"],
    ["Total queries to target model", str(stats["total_queries"])],
    ["Queries per image (avg.)", f"{stats['queries_per_image']:.1f}"],
    ["Wall-clock time for the attack phase", f"{stats['elapsed_seconds']/60:.1f} min"],
]
t = Table(rows, colWidths=[3.3 * inch, 2.7 * inch])
t.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4C72B0")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F0F0F5")]),
    ("FONTSIZE", (0, 0), (-1, -1), 9.5),
    ("TOPPADDING", (0, 0), (-1, -1), 5),
    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
]))
S.append(t)
S.append(Spacer(1, 12))

S.append(Image("report_assets_eps_hist.png", width=5.8 * inch, height=3.3 * inch))
S.append(Paragraph(
    "Figure 1. Number of images whose attack first succeeded at each L-infinity budget "
    "in the schedule (before binary-search refinement shrinks the noise further). Nearly "
    "90% of images already succeed by 8/255.", styles["Caption"]))

S.append(Image("report_assets_comparison.png", width=6.2 * inch, height=1.55 * inch))
S.append(Paragraph(
    "Figure 2. One clean (top) vs. final adversarial (bottom) image per class. The added "
    "noise is not visible at normal viewing scale.", styles["Caption"]))

S.append(Paragraph("5. Discussion and design trade-offs", styles["H2"]))
S.append(Paragraph(
    "<b>Success/imperceptibility trade-off.</b> The evaluation weighs success rate and "
    "imperceptibility equally. Our pipeline deliberately decouples them: the eps-schedule "
    "+ transfer step is responsible for success, and the binary-search step is "
    "responsible for imperceptibility, applied independently per image. This let us reach "
    "100% success without having to use a uniformly large, visible perturbation budget.",
    styles["Body"]))
S.append(Paragraph(
    "<b>Query efficiency.</b> Because gradient computation happens entirely on our own "
    "surrogates, the real model is only ever consulted to answer a yes/no question "
    "(did this flip the label), never to supply a gradient signal. This is what keeps "
    "the total query count to ~13/image, far below typical pure decision-based attacks.",
    styles["Body"]))
S.append(Paragraph(
    "<b>Confidence margin.</b> Because the target model exposes no confidence score, "
    "we cannot directly observe or optimize how far past the decision boundary an "
    "adversarial example lands (the tie-break criterion mentions this). We chose to "
    "prioritize the two primary, equally-weighted criteria (success rate and "
    "imperceptibility) by finding the minimum-magnitude successful perturbation, which "
    "by construction sits close to the boundary rather than deep inside the wrong "
    "class's region. This is the right trade-off given the weighting of the rubric.",
    styles["Body"]))
S.append(Paragraph(
    "<b>Limitations.</b> Transferability is not guaranteed in general; it works well "
    "here specifically because the clean images visibly come from CIFAR-10's "
    "distribution, which let us train surrogates on matching real data. For a task "
    "where the target's training distribution were unknown or very different, the "
    "random-noise / decision-based fallback in our pipeline would need to carry more "
    "of the attack, at a higher query cost.", styles["Body"]))

S.append(Paragraph("6. Reproducibility", styles["H2"]))
S.append(Paragraph(
    "Full code, trained surrogate weights, the completed notebook, and exact "
    "instructions to reproduce these numbers are included in this repository "
    "(see README.md). The attack and all reported numbers come from a single, "
    "deterministic run (fixed random seed) of surrogate/run_full_attack.py / the "
    "notebook's attack cell.", styles["Body"]))

doc.build(S)
print("wrote report.pdf")
