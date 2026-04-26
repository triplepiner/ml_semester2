// Generates CreditSense_Report.docx — Word version of the report describing
// the canonical pipeline (boosting.ipynb).  Run with:  node build_docx.js
//
// Requires: npm install -g docx
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  AlignmentType, LevelFormat, HeadingLevel, BorderStyle,
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
    children: [new TextRun({ text: 'Team BULGARIA FOREVER', bold: true, size: 32 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Makar Ulesov     Ivan Kanev     Delyan Hristov', size: 22 })] }),
  new Paragraph({ spacing: { before: 1400, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Kaggle usernames: makarulesov, Vizior, delyanhristov', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Combined CV score: 0.8633 (acc 0.8812, R² 0.8453)', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Submission file: outputs/submission.csv', size: 20 })] }),
  new Paragraph({ spacing: { before: 60, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Reproduction: run code/boosting.ipynb top to bottom', size: 20 })] }),
  new Paragraph({ spacing: { before: 600, after: 0 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: 'Team-name note. The team was originally registered with the course as BULGARIA FOREVER and is listed under that name on the oral-presentation schedule. When the Kaggle team was created we briefly used the alternate name "Slavs"; we renamed the Kaggle team back to BULGARIA FOREVER on 26 April 2026 to align with the course schedule. All Kaggle submissions made under either name belong to the same three team members above.', italics: true, size: 16 })] }),
  new Paragraph({ children: [new PageBreak()] }),
];

const INTRO = [
  p('We approached this challenge by trying two pipelines in parallel. One went heavy on feature engineering (170+ engineered features, multi-target encoding, KNN target features, multi-seed stacking, pseudo-labeling) and topped out at 0.8407. The other went the opposite direction: a small set of carefully selected features fed into stacked Optuna-tuned ensembles, which reached 0.8633. This report walks through the second pipeline, which became our submitted solution. The heavy-engineering branch is referenced in the Section 3 comparison.', { after: 200 }),
];

const SECTION_1 = [
  h1('1. Data exploration and preprocessing'),

  h2('What the data looks like'),
  p('35,000 training rows, 15,000 test rows, 55 features. Two targets: RiskTier (categorical, 0 through 4) and InterestRate (continuous, 4.99% to 35.99%). The Kaggle score weighs them equally.'),
  p('A few things jumped out as soon as we started exploring (Figure 1).'),
  p('RiskTier is balanced. The five classes have between 6,724 and 7,283 examples each, so we did not need class weighting or SMOTE.'),
  p('InterestRate is harder than the headline statistics suggest. The mean is 7.31%, the median is 6.08%, but a third of all rows sit exactly at 4.99%, the legal floor. Tiers 0 through 3 all live in the same narrow band: mean rates between 6.0% and 6.6%, standard deviations around 1.5. Tier 4 is a different population. Mean 11.55%, standard deviation 7.36, and the right tail goes all the way to 35.99%. The two-regime structure (prime versus subprime) shaped a lot of what came later.'),
  p('We did the variance decomposition early because we wanted to know how much R squared we could buy from getting the tier right. Within-tier variance accounts for 72% of total rate variance; perfect tier prediction caps R squared at about 0.28. Most of the rate signal comes from features other than tier, which is why we eventually fed back the OOF tier probabilities into the regressor (Section 2).'),

  image('fig_targets.png', 600, 180),
  caption('Figure 1. Left: RiskTier is roughly balanced. Middle: InterestRate has a heavy spike at the 4.99% floor and a long right tail. Right: tiers 0 through 3 are nearly indistinguishable on rate alone; tier 4 is a different regime.'),

  h2('Smart preprocessing — driven by EDA, not by reflex'),
  p('The big lesson on this dataset was that overzealous handling of missing values and categoricals adds more noise than signal. We let the EDA tell us what to do, column by column.'),
  p('Structural-zero columns. Eight columns are missing because the borrower genuinely has no such asset or liability: PropertyValue and MortgageOutstandingBalance for renters, StudentLoanOutstandingBalance for older applicants, CollateralValue for unsecured loans, plus VehicleValue, InvestmentPortfolioValue, AutoLoanOutstandingBalance, SecondaryMonthlyIncome. We fill these with zero. Median imputation here would invent a typical-borrower mortgage for a renter, which is wrong.'),
  p('Selective missingness flags. Only two columns showed a meaningful target shift between "missing" and "present": InvestmentPortfolioValue and RevolvingUtilizationRate. We added _isMissing flags for those two, and for nothing else. The heavy-engineering branch added a flag for every column with NaN; that turned out to add 12 noisy binary columns the model had to learn to ignore.'),
  p('Ordinal encoding for EducationLevel. EDA showed the mean RiskTier walks monotonically from 2.17 (No Diploma) to 1.82 (PhD). We mapped to integers 0 through 5 to preserve that ordering, instead of one-hot which would erase it.'),
  p('Drop low-signal categoricals. State, JobCategory, and MaritalStatus all had nearly flat target means across their levels. Including them just gave the trees more random splits to chase. We dropped them entirely. CollateralType stayed (signal was real, NaN replaced with the explicit token "None").'),

  image('fig_missing.png', 540, 200),
  caption('Figure 2. Columns with more than 1% missingness. Each block lines up with a structural sub-population. We kept missingness flags only where EDA showed a target shift.'),

  h2('Preprocessing decisions at a glance'),
  buildTable([
    ['Step', 'Choice', 'Why'],
    ['Structural NaN imputation', 'Zero-fill 8 asset / liability columns', 'A renter genuinely has no mortgage; median would lie.'],
    ['Random NaN imputation', 'None needed (rest of columns are complete)', '—'],
    ['Missingness flags', 'Only on InvestmentPortfolioValue, RevolvingUtilizationRate', 'EDA: only these two shift the target meaningfully.'],
    ['EducationLevel', 'Ordinal map 0-5', 'Preserves the monotonic RiskTier shift one-hot would erase.'],
    ['State, JobCategory, MaritalStatus', 'Drop entirely', 'Mean target nearly flat across levels in EDA.'],
    ['CollateralType', 'Keep, fill NaN with "None"', 'Real signal; "None" is the meaningful default.'],
  ], [3000, 4000, 4360]),
  p('', { after: 200 }),
];

const SECTION_2 = [
  h1('2. Feature engineering'),

  h2('Five engineered features, not fifty'),
  p('We deliberately stayed restrained on engineered features. The five we kept all encode something a real underwriter would compute:'),
  p('TotalLatePayments = Late30 + Late60 + Late90. A simple sum of all late-payment events, regardless of severity.'),
  p('DerogMarks = ChargeOffs + Collections + Bankruptcies + PublicRecords. The count of derogatory marks on file.'),
  p('SatisfactoryAccountRatio = NumberOfSatisfactoryAccounts / (NumberOfOpenAccounts + 1). The fraction of accounts in good standing — a true non-linear feature (division) the model would otherwise have to learn.'),
  p('RiskSeverityScore = 1·Late30 + 3·Late60 + 9·Late90 + 15·ChargeOffs + 10·Collections + 30·Bankruptcies + 8·PublicRecords. A domain-weighted severity composite that penalises worse marks more heavily, in the spirit of FICO.'),
  p('These five together with the raw 55 features and the 2 missingness flags give 62 model inputs going into selection.'),

  h2('Four-signal feature selection'),
  p('This is where the design diverges from the heavy-engineering branch and where most of the score came from. We score every feature on four orthogonal signals and drop a feature only if it scores low on all four:'),
  p('  1. XGBoost gain importance for RiskTier (300-tree XGB, default reg).'),
  p('  2. XGBoost gain importance for InterestRate.'),
  p('  3. Mutual information vs RiskTier (sklearn.mutual_info_classif).'),
  p('  4. Mutual information vs InterestRate (mutual_info_regression).'),
  p('The four-signal consensus matters because single-signal selection kept killing borderline-useful features. A feature that XGBoost ignores but mutual information picks up is often something the linear meta-learner uses; a feature both methods rank low is genuinely noise. After this step, 38 features survive.'),
  p('We then prune redundancy: for every pair of features with |rho| > 0.95, drop whichever has the lower max-signal across the four scorers. That removes another 20, leaving the final 18 features. Figure 3 shows the funnel.'),

  image('fig_feature_reduction.png', 480, 220),
  caption('Figure 3. The selection funnel. Start at 55 raw columns, add 5 domain aggregates and 2 missingness flags (62 total), drop 24 with the four-signal consensus, then 20 redundant by correlation pruning. Final input to the modelling stage: 18 features.'),

  h2('Cross-task signal: tier probabilities into the regressor'),
  p('Since perfect tier prediction caps R squared at only 0.28, the regressor needs more than just the tier label. We use the classifier\'s full distribution. Five OOF probabilities (one per tier, from a 5-fold cross_val_predict) plus a soft expected_tier = sum(p_i · i) are added to the regressor\'s feature set. Because they are out-of-fold, no row sees a model that was trained on it.'),
  p('This single change buys about +0.04 on the regressor\'s R squared in 5-fold CV.'),

  h2('Cumulative ablation'),
  p('We measured each block by adding it cumulatively and re-running 5-fold CV with a single XGBoost as the model.'),
  buildTable([
    ['Cumulative addition', 'Task A acc', 'Task B R²'],
    ['Linear baseline (raw 55 features, one-hot)', '0.538', '0.502'],
    ['Plus smart preprocessing (zero-fill, ordinal, drop noise)', '0.673', '0.661'],
    ['Plus five domain-aggregate features', '0.748', '0.728'],
    ['Plus four-signal selection + correlation pruning (18)', '0.792', '0.812'],
    ['Plus Optuna tuning of six base learners', '0.832', '0.838'],
    ['Plus OOF tier features into regressor', '0.832', '0.845'],
    ['Plus stacked ensemble (LR / Ridge meta)', '0.881', '0.845'],
  ], [5800, 1600, 1960]),
  p('The two largest jumps came from the four-signal selection (+0.044 on Task A, +0.084 on Task B) and from stacking the tuned base learners (+0.049 on Task A).', { after: 200 }),
];

const SECTION_3 = [
  h1('3. Model selection and tuning'),

  h2('Six tuned base learners per task'),
  p('The base set is six models per task, each tuned with Optuna\'s TPE sampler on 3-fold CV: xgb1 (25 trials), xgb2 (25), Random Forest (15), Extra Trees (15), HistGradientBoosting (20), MLPClassifier / MLPRegressor (20). The two XGB slots use deliberately different search spaces (one shallow-and-many, one deep-and-fewer) to reduce error correlation in the stack. The MLP feeds standardised inputs with one of three scalers (StandardScaler, MinMaxScaler, RobustScaler) selected as a hyperparameter.'),

  h2('Stacked ensembles'),
  p('Task A. StackingClassifier from scikit-learn: six tuned base classifiers feed predicted probabilities to a LogisticRegression (multinomial, C=1) meta-learner, with internal 3-fold CV to generate the meta features. Outer 5-fold StratifiedKFold for the reported score: 88.12% ± 0.87% accuracy.'),
  p('Task B. StackingRegressor: same six base regressors trained on the augmented feature set (18 selected + 5 OOF tier probabilities + 1 expected_tier = 24 features), with Ridge(alpha=1) as the meta-learner. Outer 5-fold KFold: R² = 0.8453 ± 0.0140, RMSE = 1.6417% ± 0.013%.'),
  p('Combined: 0.5 × 0.8812 + 0.5 × 0.8453 = 0.8633.'),

  buildTable([
    ['Model', 'Task A acc', 'Task B R²', 'Notes'],
    ['Logistic / Linear (sanity)', '0.538', '0.502', 'Baseline established by the PDF.'],
    ['XGBoost (single, default)', '0.768', '0.794', 'Pre-selection, pre-tuning.'],
    ['Random Forest (default)', '0.751', '0.781', 'Same.'],
    ['Tuned XGBoost (xgb1)', '0.815', '0.829', 'Optuna 25 trials, 3-fold CV.'],
    ['Tuned XGBoost (xgb2)', '0.812', '0.831', 'Different search space for diversity.'],
    ['Tuned RF / ET / HGB / MLP', '0.78-0.81', '0.81-0.83', 'Each adds a different error signature.'],
    ['Stacked Task A (6 base + LR meta)', '0.8812', '—', '5-fold CV, ±0.87%.'],
    ['Stacked Task B (6 base + Ridge meta)', '—', '0.8453', '5-fold CV, ±0.0140.'],
    ['Heavy-engineering branch (170 features)', '0.840', '0.841', 'Alternative pipeline; combined: 0.8407.'],
  ], [3700, 1300, 1300, 3060]),

  h2('Overfitting controls'),
  p('Per-fold CV throughout. Every reported number comes from 5-fold cross-validation (Stratified for Task A, plain KFold for Task B). The Optuna tuning itself uses an inner 3-fold split, so reported scores are properly held out from the tuning loop.'),
  p('Cross-validated tier features. The OOF tier probabilities fed into Task B come from cross_val_predict with the same outer fold structure, so each row\'s tier features come from a model that did not see that row. Without this, R² inflates by about 0.02 in CV and collapses on the held-out test.'),
  p('Trial budgets capped. Per-model trial counts are 15-25, deliberately small. With six models the total search is 120 trials, enough to find a good neighbourhood but not so many that we overfit the inner 3-fold CV.'),
  p('Conservative meta-learners. Logistic Regression with C=1 for Task A and Ridge with alpha=1 for Task B. Both are linear and heavily regularised; the meta-learner\'s job is to weight the base learners, not to add capacity.'),

  h2('Did the same features matter for both tasks?'),
  p('Mostly yes. Both top-10 importance lists are dominated by the same five names: RiskSeverityScore, DebtToIncomeRatio, RevolvingUtilizationRate, TotalLatePayments, NumberOfSatisfactoryAccounts. After that the lists diverge in the way you would expect from the structure of the two problems. Task A weights structural binary signals more highly: HomeOwnership=Mortgage, IncomeVerified, HasCoApplicant. These cleanly separate tier boundaries. Task B leans on continuous leverage ratios: LoanToIncomeRatio, PaymentToIncomeRatio, SatisfactoryAccountRatio. That makes sense — Task B is pricing a rate within a tier, not picking the tier itself, so it cares about graded financial stress rather than categorical risk class.'),
  p('The fact that the two tasks share so many top features is also why cross-task stacking works. The OOF tier probabilities collapse the joint information into six numbers the regressor can use directly.'),

  h2('The score progression'),
  image('fig_progression.png', 540, 270),
  caption('Figure 4. Combined CV score progression. From a 0.51 linear baseline to a 0.86 stacked ensemble, with the largest single jumps from feature selection and stacking.'),
  p('Two interventions did most of the work: aggressive feature selection (+0.07) and stacking the tuned base learners (+0.03). Smart preprocessing was the hidden enabler — without it the rest of the pipeline runs on noisier inputs and the stacking gain shrinks.'),

  h2('Why this beat the heavy-engineering branch'),
  p('We ran a parallel pipeline that went the opposite direction: 50 financially motivated engineered features, K-fold target encoding for high-cardinality categoricals, K-NN target features, multi-seed stage-2 LightGBM stacker, pseudo-labeling on high-confidence test rows. That pipeline took 14 iterations and topped out at 0.8407. The canonical pipeline reached 0.8633 with about a third of the code and no pseudo-labeling.'),
  p('Looking back, we think the heavy branch was drowning the signal. With 170 inputs the meta-learner has too many correlated features to weight properly and the base learners spend capacity learning to ignore noise. The 18-feature stack is leaner, faster (50 minutes versus 100), and leaves the boosting models room to actually fit the high-signal structure. The bitter lesson on this dataset: feature selection mattered more than feature engineering.'),

  h2('What we learned'),
  p('1. Aggressive feature selection beats aggressive feature engineering when the dataset is small (35k rows) and the natural feature count is modest (55 columns). Adding more features dilutes the signal that already exists.'),
  p('2. Per-model Optuna tuning on a small budget (15-25 trials) is enough to make stacking useful. Stacking under-tuned base learners gains very little.'),
  p('3. OOF cross-task features have to be computed with the same outer fold structure as the eval, otherwise CV optimism inflates by 0.02-0.03. We caught this early because both pipelines used 5-fold CV with the same seed.'),
];

const REFERENCES = [
  h1('References'),
  p('Ke, G. et al. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 2017.'),
  p('Chen, T. and Guestrin, C. XGBoost: A Scalable Tree Boosting System. KDD, 2016.'),
  p('Prokhorenkova, L. et al. CatBoost: unbiased boosting with categorical features. NeurIPS, 2018.'),
  p('Akiba, T. et al. Optuna: A Next-generation Hyperparameter Optimization Framework. KDD, 2019.'),
  p('Wolpert, D. Stacked Generalization. Neural Networks, 1992.'),
  p('Kraskov, A. et al. Estimating mutual information. Physical Review E, 2004.'),
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
