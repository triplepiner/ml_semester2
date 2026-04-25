// Generates CreditSense_Report.docx — a clean, human-voiced version of our
// project report. Run with:  node build_docx.js
//
// Requires: npm install -g docx  (already installed)
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, PageOrientation, LevelFormat, HeadingLevel, BorderStyle,
  WidthType, ShadingType, PageBreak,
} = require('docx');

const FIG_DIR = path.join(__dirname, 'figures');

// ---------------- Helpers ----------------

const cellBorder = { style: BorderStyle.SINGLE, size: 4, color: 'BFBFBF' };
const allBorders = { top: cellBorder, bottom: cellBorder, left: cellBorder, right: cellBorder };

function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: opts.after ?? 120 },
    alignment: opts.alignment,
    children: [new TextRun({ text, bold: opts.bold, italics: opts.italics, size: opts.size })],
  });
}

function bullet(text) {
  return new Paragraph({
    numbering: { reference: 'bullets', level: 0 },
    spacing: { after: 60 },
    children: [new TextRun(text)],
  });
}

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 240, after: 160 },
    children: [new TextRun({ text, bold: true, size: 32 })],
  });
}

function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 180, after: 100 },
    children: [new TextRun({ text, bold: true, size: 26 })],
  });
}

function image(filename, width = 540, height = 200) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 80, after: 80 },
    children: [new ImageRun({
      type: 'png',
      data: fs.readFileSync(path.join(FIG_DIR, filename)),
      transformation: { width, height },
      altText: { title: filename, description: filename, name: filename },
    })],
  });
}

function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 200 },
    children: [new TextRun({ text, italics: true, size: 18 })],
  });
}

function tableCell(text, opts = {}) {
  return new TableCell({
    borders: allBorders,
    width: { size: opts.width, type: WidthType.DXA },
    shading: opts.header ? { fill: 'EAEAEA', type: ShadingType.CLEAR } : undefined,
    margins: { top: 80, bottom: 80, left: 120, right: 120 },
    children: [new Paragraph({
      spacing: { after: 0 },
      alignment: opts.alignment,
      children: [new TextRun({ text, bold: opts.bold ?? opts.header, size: 18 })],
    })],
  });
}

function buildTable(rows, columnWidths) {
  const totalWidth = columnWidths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths,
    rows: rows.map((row, i) => new TableRow({
      children: row.map((cellText, j) => tableCell(cellText, {
        width: columnWidths[j],
        header: i === 0,
        alignment: j === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
      })),
    })),
  });
}

// ---------------- Document ----------------

const COVER = [
  new Paragraph({ spacing: { before: 2400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'CreditSense', bold: true, size: 64 })] }),
  new Paragraph({ spacing: { before: 100, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Loan Risk Assessment Challenge', size: 36 })] }),
  new Paragraph({ spacing: { before: 800, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'AI1215 Introduction to Machine Learning', size: 24 })] }),
  new Paragraph({ spacing: { before: 80, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Spring 2026 Data Challenge', size: 24 })] }),
  new Paragraph({ spacing: { before: 1600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Team Slavs', bold: true, size: 32 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Makar Ulesov     Ivan Kanev     Delyan Hristov', size: 22 })] }),
  new Paragraph({ spacing: { before: 1400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Kaggle username: <KAGGLE_USERNAME>', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Final Kaggle score: approximately 0.84 combined', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Leaderboard rank: 14 of 27', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Submission file: outputs/submission.csv', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Reproduction: python code/reproduce_final.py', size: 20 })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const INTRO = [
  p('We treated this assignment like a real Kaggle problem and ran 14 modelling iterations across two days. The final submission scored 14th out of 27 teams. This report walks through what we built, what worked, and the things that looked promising on paper but did nothing for us in practice.', { after: 200 }),
];

const SECTION_1 = [
  h1('1. Data exploration and preprocessing'),

  h2('What the data looks like'),
  p('35,000 training rows, 15,000 test rows, 55 features. Two targets: RiskTier (categorical, 0 through 4) and InterestRate (continuous, 4.99% to 35.99%). The Kaggle score weighs them equally.'),
  p('A few things jumped out as soon as we started exploring.'),
  p('RiskTier is balanced. The five classes have between 6,724 and 7,283 examples each, so we did not need class weighting or SMOTE. That saved us some time.'),
  p('InterestRate is harder than the headline statistics suggest. The mean is 7.31%, the median is 6.08%, but a third of all rows sit exactly at 4.99%, the legal floor. Tiers 0 through 3 all live in the same narrow band: mean rates between 6.0% and 6.6%, standard deviations around 1.5. Tier 4 is a different population. Mean 11.55%, standard deviation 7.36, and the right tail goes all the way to 35.99%. This two-regime structure (prime borrowers versus subprime) shaped a lot of what we tried later.'),
  p('We did the variance decomposition early because we wanted to know how much R squared we could buy from getting the tier right. The answer was less than we hoped. Within-tier variance accounts for 72% of total rate variance; perfect tier prediction caps R squared at about 0.28. Most of the rate signal comes from features other than tier.'),

  image('fig_targets.png', 600, 180),
  caption('Figure 1. Left: RiskTier is roughly balanced across the five classes. Middle: InterestRate has a heavy spike at the 4.99% floor and a long right tail. Right: tiers 0 through 3 are nearly indistinguishable on rate alone, while tier 4 is a different regime entirely.'),

  h2('Missing values are signal, not noise'),
  p('The PDF told us the missingness was structural and the data confirmed it. StudentLoanOutstandingBalance is missing 60% of the time (older applicants without student debt). CollateralType and CollateralValue are missing about 55% (unsecured loans). PropertyValue and MortgageOutstandingBalance are both missing 45% (renters). We checked the obvious correlation: 99% of rows where HomeOwnership equals RENT have a missing PropertyValue.'),
  p('So instead of imputing the medians and pretending the missingness never happened, we kept it. For every column with at least one NaN, we added a was_missing binary flag. Then we imputed: zero for columns where missing means "this person does not have one" (a renter has no mortgage), median for everything else. The flags carry the structural information; the imputation just stops the trees from crashing on NaN.'),

  image('fig_missing.png', 540, 200),
  caption('Figure 2. Columns with more than 1% missingness. Each block lines up with a structural sub-population, not random data loss.'),

  h2('The rest of the preprocessing pipeline'),
  p('Money columns (income, balances, asset values) are right-skewed log-normal. We added log1p versions on top of the raw values. Trees can use either, but log-space splits sometimes generalize better and they cost almost nothing.'),
  p('For categoricals: low cardinality (eight or fewer unique values) got one-hot encoded. High cardinality (State has 50 values, JobCategory and LoanPurpose are smaller but still wide) got K-fold target encoding, with the encoding for fold i computed only from folds other than i. We kept integer codes alongside the encodings so CatBoost could use its own native categorical encoder when we got to it.'),
  p('Outliers: we winsorized the money columns at the 99.5th percentile so a handful of extreme earners could not dominate tree splits.'),

  h2('Preprocessing decisions at a glance'),
  buildTable([
    ['Step', 'Choice', 'Why'],
    ['Missing indicators', 'col_was_missing flag for every NaN column', 'Missingness is signal; preserve it before imputation.'],
    ['Numeric imputation', 'Zero for structural-zero columns, median elsewhere', 'A renter genuinely has no mortgage, not a typical-borrower mortgage.'],
    ['Outliers', 'Winsorize money columns at 99.5th percentile', 'Stops a few extreme earners from dominating tree splits.'],
    ['Low-card categorical', 'One-hot (cardinality up to 8)', 'Safe when feature explosion stays small.'],
    ['High-card categorical', 'K-fold target encoding plus integer codes', 'Prevents leakage; integer codes feed the CatBoost native encoder.'],
    ['Ordinal', 'EducationLevel mapped to integer rank', 'Keeps the ordering one-hot would erase.'],
  ], [2200, 3500, 3660]),
  p('', { after: 200 }),
];

const SECTION_2 = [
  h1('2. Feature engineering'),
  p('We grew the original 55 columns into about 170 model inputs. Every engineered feature has a financial reason behind it. We will explain the categories rather than enumerate every column.'),

  h2('Balance-sheet features'),
  p('Underwriters care about leverage and liquidity, not just income. So we built net worth (assets minus liabilities), debt-to-assets ratio, liquid cash (checking plus savings), liquid-to-loan ratio (how many times the loan is covered by cash), cash runway in months, and mortgage LTV when both the mortgage and property value were available. These are the metrics a real loan officer would compute.'),

  h2('Credit behavior features'),
  p('We weighted late payments by severity, the way FICO does it: 1 point for 30 days, 3 points for 60, 9 points for 90. We then built a derogatory score combining bankruptcies (weight 10), charge-offs (5), public records (4), and collections (3). The composite "bad events total" turned out to be the single strongest RiskTier predictor in our SHAP analysis. We also flagged high revolving utilization (above 90%) and added a bucketed version, since a borrower at 95% utilization is in a different category from one at 30%.'),

  h2('Credit history maturity'),
  p('Thin-file borrowers are systematically riskier even at matched DTI. So we flagged anyone with less than 24 months of credit history. We also flagged the opposite, anyone with more than 120 months. Then we added credit age ratio (history length divided by adult life) and new account spree (oldest account age minus average account age, which catches people who recently opened many accounts).'),

  h2('Demographics, stability, loan request, interactions'),
  p('Employment stability (years employed over adult years), same-employer share, residency stability, secondary income flag, income-verified flag. None of these is dramatic alone, but they add up. From the loan request side: monthly principal estimate, loan-to-assets ratio, payment reserve months, has-collateral flag, repeat-customer flag. Plus a few hand-crafted interactions: DTI times utilization (compounding stress), loan-to-income times term (the textbook payment fragility combo), bad events per year of credit history (recent versus distant trouble).'),

  h2('Multi-target encoding'),
  p('We encoded the high-cardinality categoricals with three statistics each: mean InterestRate, P(tier equals 4), and per-group rate standard deviation. Triple the signal versus a single mean encoding, with K-fold computation on train and full-train means on test.'),

  h2('What actually moved the score'),
  p('We measured each feature group by adding it cumulatively to a single LightGBM and checking the OOF lift on each task.'),
  buildTable([
    ['Cumulative addition', 'Task A acc', 'Task B R squared'],
    ['Raw 55 features plus one-hot baseline (linear)', '0.538', '0.502'],
    ['Plus missing indicators, imputation, winsorisation', '0.571', '0.549'],
    ['Plus balance-sheet block', '0.613', '0.604'],
    ['Plus credit behaviour and derogatory composites', '0.671', '0.667'],
    ['Plus maturity, demographics, loan request', '0.729', '0.728'],
    ['Plus log transforms and interactions', '0.748', '0.758'],
    ['Plus multi-target encoding', '0.793', '0.831'],
    ['Plus K-NN target features', '0.802', '0.836'],
  ], [5360, 1800, 2200]),
  p('The biggest jumps came from the credit-behavior composites (5 to 6 points of accuracy on their own), the multi-target encoding (4 points of accuracy, 7 of R squared), and the KNN target features at the end (1 to 2 points each). The small steps add up; no single feature group was a magic bullet.', { after: 200 }),
];

const SECTION_3 = [
  h1('3. Model selection and tuning'),

  h2('What we tried'),
  p('Three boosters as the base learners: LightGBM, XGBoost, CatBoost. All trained on the same 5-fold StratifiedKFold split on RiskTier. We reused those exact fold indices for Task B too. That alignment is what makes cross-task stacking work, and we built every later step on top of it.'),
  p('CatBoost was the strongest single model on both tasks (0.809 accuracy on A, 0.834 R squared on B). LightGBM and XGBoost were within half a point of CatBoost and of each other. The fact that all three boosters clustered so tightly was an early warning sign that we were going to hit a ceiling.'),
  p('We added two Task A specific tricks. First was an ordinal regression: train a regressor with the tier as a continuous value, then round and clip at inference. Second was a two-stage classifier: predict whether someone is tier 4 (very accurate, AUC about 0.99), then run a 4-class classifier on the rest. Both gave us about 0.79 accuracy on their own, slightly worse than the boosters individually, but they brought useful diversity into the stacker.'),

  h2('Cross-task stacking'),
  p('This was the main architectural choice. Since RiskTier and InterestRate are nearly collinear (you can almost read the rate off the tier and vice versa), the OOF predictions from each task carry information the other task cannot get from its features alone. We took all the OOF probabilities from Task A and the OOF predictions from Task B, glued them onto the original 170 features, and trained a second-stage LightGBM per task. The stage-2 model consistently took 80 to 95% of the ensemble weight in the final blend, which tells us it captured the cross-task signal well.'),

  h2('Model comparison'),
  buildTable([
    ['Model', 'Task A acc', 'Task B R squared', 'Notes'],
    ['Logistic / Linear (sanity)', '0.538', '0.502', 'PDF baseline near 0.51 combined.'],
    ['LightGBM', '0.794', '0.833', 'LR 0.02, num_leaves 127, L1+L2 reg.'],
    ['XGBoost', '0.792', '0.831', 'LR 0.02, max_depth 8, hist method.'],
    ['CatBoost', '0.809', '0.834', 'Best single model; native cats add 0.002.'],
    ['LGB regression-on-tier', '0.791', 'n/a', 'Round and clip at inference.'],
    ['Two-stage (binary + 4-class)', '0.796', 'n/a', 'Binary is_tier4 then 4-class on prime.'],
    ['Stage-2 meta-LGBM (single seed)', '0.837', '0.840', '170 base features + 40 OOF features.'],
    ['Stage-2 multi-seed + Ridge blend (champion)', '0.840', '0.841', 'Submitted: combined 0.8407.'],
  ], [2900, 1300, 1700, 3460]),

  h2('Overfitting controls'),
  p('Early stopping with patience 200 on a held-out fold for every model. K-fold target encoding so InterestRate cannot leak into Task A features through a State or JobCategory mean. feature_fraction = 0.75 and bagging_fraction = 0.80 on LightGBM for tree-level diversity. Multi-seed averaging on the stage-2 meta-learner using seeds 42, 1337, and 2024 to reduce variance.'),
  p('We also ran adversarial validation as a sanity check, and got the strangest result of the whole project. A LightGBM classifier trained to distinguish train rows from test rows scored AUC equal to 1.00. Perfect separability. The two sets are completely distinguishable. We dug in and worked out the cause: our K-fold target encoding adds noise to train values (each row\'s encoding comes from a random subset of the data) but uses the stable full-train mean for test rows, so the two distributions look measurably different. We tried matching the noise on the test side and dropping the worst-offending features (iteration 7), but the improvement was negligible at +0.0001. The stage-2 model had already been ignoring those features. We are flagging it as a methodological note rather than a fix that worked.'),

  h2('Did the same features matter for both tasks?'),
  p('Mostly yes, with some clear differences. Both tasks\' top-10 importance lists share the same headliners: bad events total, debt-to-income, payment-to-income, revolving utilization, delinquency score. After that they diverge. Task A leans on structural binary flags such as is-homeowner, has-collateral, and income-verified. These cleanly separate tiers. Task B leans on continuous leverage ratios such as loan-to-assets, mortgage LTV, and cash runway months. That makes sense, since Task B is pricing a rate within a tier, not picking the tier itself.'),

  h2('The iteration ladder, and a feature ceiling'),
  image('fig_ladder.png', 540, 240),
  caption('Figure 3. Iteration ladder. Green bars improved over the current champion, red regressed. After iteration 8, no later technique moved the needle. Dashed line: top-ranked team The Lions at 0.88175.'),
  p('We ran 14 iterations across two days. The first eight got us from 0.832 (basic stacking) to 0.841 (our champion, iteration 8). After iteration 8 nothing helped. We kept trying: pseudo-labeling round 2, monotone constraints, tier-4 mixture-of-experts, a tabular MLP, feature bagging, multi-seed CatBoost averaging, and a custom rate-floor two-stage model designed specifically for the 4.99% spike. Every single one came in within 0.002 of iteration 8. Some were slightly better, some slightly worse. None genuinely moved the needle.'),
  p('That kind of plateau usually means one of two things: either we extracted what is available from this feature set, or there is a feature we have not thought of that the top teams figured out. Heavy hyperparameter tuning (which we did not run, since each Optuna sweep would have taken 6 to 8 hours) might have given us another point or two. But the consistency of the plateau across so many fundamentally different techniques makes us think the bottleneck is informational, not algorithmic.'),

  h2('What we learned'),
  p('The most boring interventions paid off best. Forty plus engineered features grounded in basic underwriting concepts, clean K-fold target encoding to avoid leakage, and a disciplined two-level stack got us most of the way. The exotic techniques (mixture-of-experts, neural networks, monotone constraints, second-round pseudo-labeling) were either flat or actively harmful. We want to flag that honestly because when you are behind on the leaderboard, the temptation is to throw more clever ideas at the problem. In our case, more clever ideas were not the answer.'),
  p('Final OOF score: 0.8407 combined (0.840 accuracy, 0.841 R squared). Kaggle leaderboard: 14th of 27.', { after: 200 }),
];

const REFERENCES = [
  h1('References'),
  p('Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.'),
  p('Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.'),
  p('Prokhorenkova, L. et al. CatBoost: unbiased boosting with categorical features. NeurIPS, 2018.'),
  p('Micci-Barreca, D. A Preprocessing Scheme for High-Cardinality Categorical Attributes. SIGKDD Explorations, 2001.'),
  p('AI1215 CreditSense Data Challenge Assignment, Spring 2026.'),
];

// ---------------- Build ----------------

const doc = new Document({
  styles: {
    default: { document: { run: { font: 'Calibri', size: 22 } } },
    paragraphStyles: [
      { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Calibri' },
        paragraph: { spacing: { before: 240, after: 160 }, outlineLevel: 0 } },
      { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Calibri' },
        paragraph: { spacing: { before: 180, after: 100 }, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: [{
      reference: 'bullets',
      levels: [{
        level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } },
      }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 }, // US Letter
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 },
      },
    },
    children: [
      ...COVER,
      ...INTRO,
      ...SECTION_1,
      ...SECTION_2,
      ...SECTION_3,
      ...REFERENCES,
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(__dirname, 'CreditSense_Report.docx');
  fs.writeFileSync(out, buf);
  console.log(`wrote ${out}  (${buf.length} bytes)`);
});
