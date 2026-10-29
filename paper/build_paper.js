// Builds the term paper (.docx) from the experiment results.
// Usage: node build_paper.js <projectRoot> <outFile>
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, WidthType,
  AlignmentType, HeadingLevel, BorderStyle, ShadingType, Footer, PageNumber, LevelFormat,
  TabStopType, Math: OMath, MathRun, MathFraction, MathRadical, MathSubSuperScript, MathSubScript,
  VerticalAlign,
} = require("docx");

const ROOT = process.argv[2];
const OUT = process.argv[3];
const RES = path.join(ROOT, "research", "results");
const FIG = path.join(ROOT, "research", "figures");

// ---------------------------------------------------------------- data
function splitCsvLine(line) {
  // Handles double-quoted fields that contain commas (as written by pandas)
  const out = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (q) {
      if (ch === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (ch === '"') q = false;
      else cur += ch;
    } else if (ch === '"') q = true;
    else if (ch === ",") { out.push(cur); cur = ""; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}
function readCsv(name) {
  const lines = fs.readFileSync(path.join(RES, name), "utf8").trim().split(/\r?\n/);
  const head = splitCsvLine(lines[0]);
  return lines.slice(1).map((l) => {
    const v = splitCsvLine(l);
    const o = {};
    head.forEach((h, i) => { const n = Number(v[i]); o[h] = v[i] !== "" && !isNaN(n) ? n : v[i]; });
    return o;
  });
}
const summary = JSON.parse(fs.readFileSync(path.join(RES, "summary.json"), "utf8"));
const mlops = JSON.parse(fs.readFileSync(path.join(RES, "e5_mlops.json"), "utf8"));
const e1 = readCsv("e1_outliers.csv");
const e1r = readCsv("e1_rare_houses.csv");
const e2 = readCsv("e2_ladder.csv");
const e3 = readCsv("e3_models.csv");
const sig = readCsv("e3_significance_all.csv");
const imp = readCsv("e4_importance.csv");

// ---------------------------------------------------------------- formatting helpers
const FONT = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1440, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, 1 inch margins

const sup = (n) => String(n).split("").map((d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d] || (d === "-" ? "⁻" : d)).join("");
function sci(v, prefix = "") {
  const e = Math.floor(Math.log10(Math.abs(v)));
  return `≈ ${prefix}${(v / 10 ** e).toFixed(1)} × 10${sup(e)}`;
}
const usd = (v) => (Math.abs(v) >= 1e7 ? sci(v, "$") : "$" + Math.round(v).toLocaleString("en-US"));
const num = (v, d = 3) => (Math.abs(v) >= 1e5 ? sci(v) : v.toFixed(d).replace("-", "−"));
const pm = (m, s, d = 3) => `${num(m, d)} ± ${num(s, d)}`;
const pval = (p) => (p < 0.001 ? "< 0.001" : p.toFixed(3));

// ---------------------------------------------------------------- citations (numbered by first use)
const REFS = {
  rosen: 'S. Rosen, "Hedonic prices and implicit markets: Product differentiation in pure competition," *Journal of Political Economy*, vol. 82, no. 1, pp. 34–55, 1974.',
  decock: 'D. De Cock, "Ames, Iowa: Alternative to the Boston housing data as an end of semester regression project," *Journal of Statistics Education*, vol. 19, no. 3, 2011, doi: 10.1080/10691898.2011.11889627.',
  sculley: 'D. Sculley *et al.*, "Hidden technical debt in machine learning systems," in *Advances in Neural Information Processing Systems 28 (NeurIPS)*, 2015, pp. 2503–2511.',
  kreuzberger: 'D. Kreuzberger, N. Kühl, and S. Hirschl, "Machine learning operations (MLOps): Overview, definition, and architecture," *IEEE Access*, vol. 11, pp. 31866–31879, 2023, doi: 10.1109/ACCESS.2023.3262138.',
  zenml: 'ZenML, "ZenML: An open-source MLOps framework," version 0.64.0, 2024. [Online]. Available: https://zenml.io',
  mlflow: 'M. Zaharia *et al.*, "Accelerating the machine learning lifecycle with MLflow," *IEEE Data Engineering Bulletin*, vol. 41, no. 4, pp. 39–45, 2018.',
  kaufman: 'S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, "Leakage in data mining: Formulation, detection, and avoidance," *ACM Transactions on Knowledge Discovery from Data*, vol. 6, no. 4, Art. 15, 2012.',
  kapoor: 'S. Kapoor and A. Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science," *Patterns*, vol. 4, no. 9, Art. 100804, 2023, doi: 10.1016/j.patter.2023.100804.',
  orig: 'A. Singh, "prices-predictor-system: An end-to-end house price prediction pipeline with ZenML and MLflow," source code. [Online]. Available: [[insert the URL you downloaded the project from]]',
  sharma: 'H. Sharma, H. Harsora, and B. Ogunleye, "An optimal house price prediction algorithm: XGBoost," *Analytics*, vol. 3, no. 1, pp. 30–45, 2024.',
  xgb: 'T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.',
  rf: 'L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.',
  ridge: 'A. E. Hoerl and R. W. Kennard, "Ridge regression: Biased estimation for nonorthogonal problems," *Technometrics*, vol. 12, no. 1, pp. 55–67, 1970.',
  lasso: 'R. Tibshirani, "Regression shrinkage and selection via the lasso," *Journal of the Royal Statistical Society: Series B*, vol. 58, no. 1, pp. 267–288, 1996.',
  nadeau: 'C. Nadeau and Y. Bengio, "Inference for the generalization error," *Machine Learning*, vol. 52, no. 3, pp. 239–281, 2003.',
  gamma: 'E. Gamma, R. Helm, R. Johnson, and J. Vlissides, *Design Patterns: Elements of Reusable Object-Oriented Software*. Reading, MA, USA: Addison-Wesley, 1994.',
  sklearn: 'F. Pedregosa *et al.*, "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.',
  fisher: 'A. Fisher, C. Rudin, and F. Dominici, "All models are wrong, but many are useful: Learning a variable\'s importance by studying an entire class of prediction models simultaneously," *Journal of Machine Learning Research*, vol. 20, no. 177, pp. 1–81, 2019.',
};
const order = [];
function c(...keys) {
  return keys.map((k) => {
    if (!REFS[k]) throw new Error("unknown ref " + k);
    if (!order.includes(k)) order.push(k);
    return `[${order.indexOf(k) + 1}]`;
  }).join(", ");
}

// ---------------------------------------------------------------- text helpers
// Markup: **bold**, *italic*, [[placeholder]] (yellow highlight)
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\[\[[^\]]+\]\]|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const t = m[0];
    if (t.startsWith("_{")) out.push(new TextRun({ text: t.slice(2, -1), subScript: true, font: FONT, ...base }));
    else if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (t.startsWith("[[")) out.push(new TextRun({ text: "[" + t.slice(2, -2) + "]", highlight: "yellow", font: FONT, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, font: FONT, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}
const P = (text, opts = {}) => new Paragraph({
  children: runs(text, opts.run || {}),
  alignment: opts.align || AlignmentType.JUSTIFIED,
  spacing: { after: opts.after ?? 120, line: 276 },
  indent: opts.indent,
  numbering: opts.numbering,
  keepNext: opts.keepNext,
});
const H1 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun({ text, font: FONT })] });
const H2 = (text) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun({ text, font: FONT })] });
const NUM = (text, ref = "contrib") => P(text, { numbering: { reference: ref, level: 0 }, after: 60 });

// ---------------------------------------------------------------- figures
let figNo = 0;
function pngSize(file) {
  const b = fs.readFileSync(file);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function figure(file, widthIn, caption) {
  figNo += 1;
  const f = path.join(FIG, file);
  const { w, h } = pngSize(f);
  const wpx = widthIn * 96;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      children: [new ImageRun({ type: "png", data: fs.readFileSync(f), transformation: { width: wpx, height: (wpx * h) / w } })],
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Fig. ${figNo}. `, font: FONT, size: 18, bold: true }), ...runs(caption, { size: 18 })],
    }),
  ];
}
function figurePlaceholder(caption, note) {
  figNo += 1;
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 120, after: 60 },
      border: ["top", "bottom", "left", "right"].reduce((o, s) => ({ ...o, [s]: { style: BorderStyle.DASHED, size: 6, color: "999999", space: 12 } }), {}),
      children: runs(`[[${note}]]`),
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER, spacing: { after: 200 },
      children: [new TextRun({ text: `Fig. ${figNo}. `, font: FONT, size: 18, bold: true }), ...runs(caption, { size: 18 })],
    }),
  ];
}

// ---------------------------------------------------------------- tables
let tabNo = 0;
const ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII"];
const border = { style: BorderStyle.SINGLE, size: 4, color: "808080" };
const borders = { top: border, bottom: border, left: border, right: border };
function table(title, headers, rows, fracs, opts = {}) {
  tabNo += 1;
  const widths = fracs.map((f) => Math.floor(CONTENT_W * f));
  widths[widths.length - 1] = CONTENT_W - widths.slice(0, -1).reduce((a, b) => a + b, 0);
  const align = opts.align || headers.map((_, i) => (i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER));
  const cell = (text, i, header) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders,
    verticalAlign: VerticalAlign.CENTER,
    shading: header ? { type: ShadingType.CLEAR, fill: "EDEDED", color: "auto" } : undefined,
    margins: { top: 40, bottom: 40, left: 80, right: 80 },
    children: [new Paragraph({ alignment: header ? AlignmentType.CENTER : align[i], children: runs(String(text), { size: 17, bold: header || undefined }) })],
  });
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 200, after: 0 },
      children: [new TextRun({ text: `TABLE ${ROMAN[tabNo - 1]}`, font: FONT, size: 18 })] }),
    new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { after: 80 },
      children: [new TextRun({ text: title.toUpperCase(), font: FONT, size: 16 })] }),
    new Table({
      width: { size: CONTENT_W, type: WidthType.DXA },
      columnWidths: widths,
      rows: [
        new TableRow({ tableHeader: true, cantSplit: true, children: headers.map((h, i) => cell(h, i, true)) }),
        ...rows.map((r) => new TableRow({ cantSplit: true, children: r.map((v, i) => cell(v, i, false)) })),
      ],
    }),
    ...(opts.note ? [new Paragraph({ spacing: { before: 60, after: 200 }, children: runs(opts.note, { size: 16, italics: true }) })]
                  : [new Paragraph({ spacing: { after: 120 }, children: [] })]),
  ];
}

// ---------------------------------------------------------------- equation
function equation() {
  const mr = (t) => new MathRun(t);
  const sigma2 = new MathSubSuperScript({ children: [mr("s")], subScript: [mr("d")], superScript: [mr("2")] });
  const mean = new MathSubScript({ children: [mr("μ")], subScript: [mr("d")] });
  const n2 = new MathSubScript({ children: [mr("n")], subScript: [mr("2")] });
  const n1 = new MathSubScript({ children: [mr("n")], subScript: [mr("1")] });
  return new Paragraph({
    tabStops: [{ type: TabStopType.CENTER, position: CONTENT_W / 2 }, { type: TabStopType.RIGHT, position: CONTENT_W }],
    spacing: { before: 120, after: 120 },
    children: [
      new TextRun("\t"),
      new OMath({ children: [
        mr("t = "),
        new MathFraction({
          numerator: [mean],
          denominator: [new MathRadical({ children: [
            mr("("), new MathFraction({ numerator: [mr("1")], denominator: [mr("J")] }), mr(" + "),
            new MathFraction({ numerator: [n2], denominator: [n1] }), mr(") "), sigma2,
          ] })],
        }),
      ] }),
      new TextRun({ text: "\t(1)", font: FONT }),
    ],
  });
}

// ---------------------------------------------------------------- numbers used in the text
const E0 = summary.exp0;
const nRare = summary.houses_flagged_by_global_z;
const pctRare = (100 * nRare / summary.rows).toFixed(1);
const r1 = (rule, proto) => e1.find((r) => r.rule === rule && r.protocol === proto);
const g = r1("Global Z-score (original)", "Filter training folds only");
const gA = r1("Global Z-score (original)", "Filter before split");
const tB = r1("Target Z-score", "Filter training folds only");
const dA = r1("De Cock (>4000 sq ft)", "Filter before split");
const dB = r1("De Cock (>4000 sq ft)", "Filter training folds only");
const rare = (rule, grp) => e1r.find((r) => r.rule === rule && r.group === grp);
const gr = rare("Global Z-score (original)", "Houses with a rare feature");
const go = rare("Global Z-score (original)", "Other houses");
const tr = rare("Target Z-score", "Houses with a rare feature");
const m3 = (model, f) => e3.find((r) => r.model === model && r.features === f);
const NC = "Numeric + categorical", NO = "Numeric only";
const best = m3("XGBoost", NC);
const s3 = (model, f) => sig.find((r) => r.model === model && r.features === f);
const ho = summary.exp4_holdout;
const L = (prefix) => e2.find((r) => r.step.startsWith(prefix));

// ---------------------------------------------------------------- content
const body = [];
const add = (...xs) => xs.flat().forEach((x) => body.push(x));

// Title block
add(
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "When Data Cleaning Inflates Accuracy: A Reproducible MLOps Pipeline and Leakage-Aware Evaluation for House Price Prediction", font: FONT, size: 36 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: runs("[[Your Full Name]]  ([[Roll No.]])", { size: 22 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: runs("Department of [[Branch, e.g. Computer Engineering]], [[College Name]], [[City]], India", { size: 20, italics: true }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 0 }, children: runs("[[your.email@college.edu]]", { size: 20 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 0 }, children: runs("Term paper — [[Course name and code]], Semester VII, [[Academic year]]", { size: 20 }) }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 320 }, children: runs("Guide: [[Guide's name and designation]]", { size: 20 }) }),
);

// Abstract
add(
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 264 }, children: [
    new TextRun({ text: "Abstract", bold: true, italics: true, font: FONT, size: 19 }),
    new TextRun({ text: "—", bold: true, font: FONT, size: 19 }),
    ...runs(`Machine learning projects for house price prediction often report high accuracy, but preprocessing decisions taken before the train–test split can make these numbers misleading. This paper studies an open-source, end-to-end house price prediction system built with ZenML and MLflow on the Ames, Iowa housing dataset (${summary.rows.toLocaleString("en-US")} sales). The original pipeline reported a coefficient of determination (R²) of ${E0.original.R2_log.toFixed(3)} for a linear regression model. We show that this figure is an artefact of its outlier-removal step, which applied a z-score rule to every numeric column of the full dataset and silently discarded ${nRare} houses (${pctRare}%) that have ordinary prices but rare features such as a basement half-bath or a second kitchen. Under repeated 5-fold cross-validation in which outliers are removed from the training folds only and every house is tested, the same model achieves R² = ${pm(g.R2_log_mean, g.R2_log_std, 2)}, and its mean absolute percentage error (MAPE) on houses with rare features rises to ${gr.MAPE_pct_mean.toFixed(1)}%. Restricting outlier detection to the target variable restores R² = ${pm(tB.R2_log_mean, tB.R2_log_std)}. With a corrected, leakage-aware protocol, XGBoost using numeric and categorical features performs best (R² = ${pm(best.R2_log_mean, best.R2_log_std)}, MAPE = ${best.MAPE_pct_mean.toFixed(1)}%), although its advantage over regularised linear models is not statistically significant. We also make the pipeline portable to Windows, where ZenML's MLflow model deployer is unsupported, and measure its MLOps behaviour: identical evaluation scores across repeated runs and a median prediction latency of ${mlops.latency_single.median_ms.toFixed(0)} ms.`, { size: 19, bold: true }),
  ] }),
  new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 240 }, children: [
    new TextRun({ text: "Index Terms", bold: true, italics: true, font: FONT, size: 19 }),
    new TextRun({ text: "—House price prediction, MLOps, data leakage, outlier detection, cross-validation, XGBoost, reproducibility, ZenML, MLflow.", bold: true, font: FONT, size: 19 }),
  ] }),
);

// I. Introduction
add(H1("I. INTRODUCTION"),
  P(`A house is the largest asset most households own, and automated valuation models help buyers, lenders and tax assessors price it consistently. Hedonic pricing theory treats a property's price as the sum of the implicit prices of its characteristics, such as size, quality and location ${c("rosen")}. Machine learning extends this idea with flexible models, and the Ames, Iowa housing dataset has become a standard benchmark for it ${c("decock")}.`),
  P(`Machine learning systems are increasingly delivered as automated pipelines rather than one-off notebooks. This practice, known as machine learning operations (MLOps), addresses the hidden engineering costs of production machine learning ${c("sculley", "kreuzberger")}. Tools such as ZenML ${c("zenml")} orchestrate pipeline steps and version their outputs, while MLflow ${c("mlflow")} tracks experiments and serves models. Automation, however, does not guarantee that an evaluation is valid. Data leakage, where information that would not be available at prediction time influences training or evaluation, is common and inflates reported performance ${c("kaufman", "kapoor")}.`),
  P(`This paper is a case study of an open-source house price prediction pipeline ${c("orig")}. While adapting it to run on Windows, we noticed that the deployed model produced absurd prices for some ordinary houses; one typical 1976 house was valued at $1,375. We traced the problem to the pipeline's outlier-removal step, which also explains its high reported accuracy. We therefore ask three research questions:`),
  P(`**RQ1:** How much do data-cleaning choices, in particular outlier removal and where it is applied, change the reported accuracy?`, { indent: { left: 360 }, after: 40 }),
  P(`**RQ2:** Which regression models perform best once the evaluation is free of leakage, and are the differences statistically significant?`, { indent: { left: 360 }, after: 40 }),
  P(`**RQ3:** How well does the MLOps tooling support reproducible results and deployment across operating systems?`, { indent: { left: 360 } }),
  P(`Our contributions are:`, { after: 60 }),
  NUM(`An exact reproduction of the original reported result (R² = ${E0.original.R2_log.toFixed(4)}) and a demonstration that it is inflated because the test set was filtered; evaluated on all houses, the same model scores R² = ${num(g.R2_log_mean, 2)}.`),
  NUM(`A controlled comparison of five outlier-removal strategies under two evaluation protocols, and a step-by-step ablation from the original pipeline to a corrected one.`),
  NUM(`A leakage-aware comparison of five regression models with and without categorical features, using a corrected resampled t-test.`),
  NUM(`A Windows-compatible deployment path for the pipeline, with measurements of reproducibility, run time and prediction latency.`),
  P(`Section II reviews related work, Section III describes the dataset and Section IV the pipeline and evaluation method. Section V presents the results, Section VI discusses them, Section VII lists limitations and Section VIII concludes.`, { after: 200 }),
);

// II. Related work
add(H1("II. RELATED WORK"),
  H2("A. House Price Prediction"),
  P(`Hedonic regression ${c("rosen")} remains the conceptual basis for house price models. De Cock introduced the Ames dataset of ${summary.rows.toLocaleString("en-US")} residential sales from 2006 to 2010 as a richer alternative to the Boston housing data, and recommended removing the few houses larger than 4,000 sq ft before modelling ${c("decock")}. Sharma *et al.* compared XGBoost, support vector regression, random forest, multilayer perceptron and multiple linear regression on the same data and found XGBoost to perform best ${c("sharma")}. Gradient-boosted trees ${c("xgb")}, random forests ${c("rf")} and regularised linear models such as ridge ${c("ridge")} and lasso ${c("lasso")} are the standard baselines for this kind of tabular regression.`),
  H2("B. Data Leakage and Evaluation Pitfalls"),
  P(`Kaufman *et al.* formalised leakage in data mining and showed how it produces over-optimistic models ${c("kaufman")}. Kapoor and Narayanan surveyed machine-learning-based science and found leakage affecting at least 294 papers across 17 fields; their taxonomy of eight leakage types includes preprocessing that is fitted on training and test data together ${c("kapoor")}. When models are compared with repeated cross-validation, the training sets overlap, so the standard paired t-test is too optimistic; Nadeau and Bengio proposed a corrected variance estimate for this setting ${c("nadeau")}.`),
  H2("C. MLOps"),
  P(`Sculley *et al.* argued that most of the cost of production machine learning lies outside the model code, in data dependencies, configuration and pipelines ${c("sculley")}. Kreuzberger *et al.* defined MLOps through principles such as automation, versioning, reproducibility and experiment tracking ${c("kreuzberger")}. MLflow implements experiment tracking, model packaging and serving ${c("mlflow")}, and ZenML composes such tools into portable pipelines ${c("zenml")}. These works focus on engineering; this paper shows how a pipeline can faithfully automate an evaluation that is itself flawed.`, { after: 200 }),
);

// III. Dataset
const ds = summary;
add(H1("III. DATASET"),
  P(`The Ames Housing dataset contains ${ds.rows.toLocaleString("en-US")} residential property sales in Ames, Iowa, between 2006 and 2010, recorded by the City Assessor's Office ${c("decock")}. Each sale has ${ds.columns} columns: two identifiers (Order and PID), the target SalePrice, 36 numeric predictors and ${ds.categorical_columns} categorical predictors (Table I). There are ${ds.missing_cells.toLocaleString("en-US")} missing cells in ${ds.columns_with_missing} columns. In most categorical columns a missing value means that the feature is absent, for example a house without a pool or a garage.`),
  P(`Sale prices range from $${ds.price_min.toLocaleString("en-US")} to $${ds.price_max.toLocaleString("en-US")}, with a median of $${ds.price_median.toLocaleString("en-US")}. The distribution is right-skewed (skewness ${ds.price_skew.toFixed(2)}); after the transformation log(1 + price) the skewness is ${num(ds.log_price_skew, 2)}. All models are therefore trained to predict log price, and predictions are converted back to dollars for evaluation.`),
  table("Summary of the Ames Housing dataset", ["Property", "Value"], [
    ["Sales (rows)", ds.rows.toLocaleString("en-US")],
    ["Period", "2006–2010"],
    ["Columns", `${ds.columns} (2 identifiers, 1 target, 36 numeric, ${ds.categorical_columns} categorical)`],
    ["Missing cells", `${ds.missing_cells.toLocaleString("en-US")} in ${ds.columns_with_missing} columns`],
    ["Sale price: min / median / mean / max", `$${ds.price_min.toLocaleString("en-US")} / $${ds.price_median.toLocaleString("en-US")} / $${Math.round(ds.price_mean).toLocaleString("en-US")} / $${ds.price_max.toLocaleString("en-US")}`],
    ["Skewness of price / of log(1 + price)", `${ds.price_skew.toFixed(2)} / ${num(ds.log_price_skew, 2)}`],
  ], [0.42, 0.58]),
);

// IV. Methodology
add(H1("IV. METHODOLOGY"),
  H2("A. Pipeline Architecture"),
  P(`Fig. 1 shows the system. A ZenML training pipeline runs seven steps: (1) ingestion of the dataset from a ZIP archive; (2) filling missing numeric values with the column mean; (3) a log(1 + x) transform of living area (Gr Liv Area) and SalePrice; (4) outlier removal; (5) an 80/20 train–test split with a fixed random seed; (6) training of a scikit-learn ${c("sklearn")} pipeline consisting of a mean imputer, one-hot encoding of categorical columns and linear regression; and (7) evaluation by mean squared error (MSE) and R² on the log scale. ZenML versions every step's output, and MLflow autologging records parameters, metrics and the fitted model. The code uses the factory, strategy and template-method design patterns ${c("gamma")}, so that, for example, a different outlier or missing-value strategy can be swapped in without changing the pipeline.`),
  ...figure("fig1_architecture.png", 6.2, "Architecture of the house price prediction system. The training pipeline (top) is orchestrated by ZenML; MLflow tracks each run and serves the trained model, which a batch inference pipeline queries over HTTP."),
  P(`On Linux and macOS, ZenML's MLflow model deployer starts the MLflow scoring server as a background daemon. ZenML does not support daemons on Windows, so the original deployment script fails there. We added a Windows path: the deployment script detects the operating system, retrieves the newest model from the MLflow tracking store, launches the MLflow scoring server as a detached process, waits for its health check and then runs a ZenML inference pipeline that sends batch requests over HTTP. We also replaced a hard-coded absolute data path and fixed the inference scripts, which sent raw living area to a model trained on log living area and reported the log-scale output as a price.`),
  H2("B. The Original Outlier Step"),
  P(`The original outlier step computes z = |x − μ| / σ for every numeric column of the full dataset, including the identifiers and the log target, and drops a row if any column has z > 3. Although the step is given the argument column_name = "SalePrice", it uses that argument only for input validation. The step runs before the train–test split, so the test set is filtered too. It also keeps only numeric columns, so the model never sees categorical information such as neighbourhood or quality ratings.`),
  P(`This rule removed ${nRare} of ${ds.rows.toLocaleString("en-US")} houses (${pctRare}%). Table II lists the columns that triggered it most often. Many numeric features are zero for most houses, so any non-zero value lies more than three standard deviations from the mean: every house with a basement half-bath, a second kitchen or a screened porch was removed. The removed houses are not price outliers; their median price was $159,000, against $161,875 for the houses that were kept.`),
  table("Columns that most often triggered the original outlier rule", ["Column", "Houses flagged", "Meaning"], [
    ["Bsmt Half Bath", "175", "Half bathrooms in the basement"],
    ["Kitchen AbvGr", "134", "Kitchens above ground (usually 2 instead of 1)"],
    ["Screen Porch", "107", "Screened porch area (sq ft)"],
    ["BsmtFin SF 2", "92", "Second finished basement area (sq ft)"],
    ["Enclosed Porch", "88", "Enclosed porch area (sq ft)"],
    ["Mas Vnr Area", "63", "Masonry veneer area (sq ft)"],
  ], [0.25, 0.2, 0.55], { align: [AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.LEFT], note: `A house can be flagged by several columns; ${nRare} distinct houses were removed in total.` }),
  H2("C. Evaluation Protocol"),
  P(`All experiments except the reproduction in Section V-A use 5-fold cross-validation repeated 3 times with seed ${ds.seed}, giving J = 15 train/test splits. Every data-dependent step (imputation, scaling, encoding and outlier thresholds) is fitted on the training fold only, using scikit-learn pipelines ${c("sklearn")}. Outlier removal is evaluated under two protocols:`),
  P(`**Protocol A (filter before split)**, as in the original pipeline: the rule is applied to the whole dataset, so the test folds are filtered as well.`, { indent: { left: 360 }, after: 40 }),
  P(`**Protocol B (filter training folds only)**: the rule is fitted on and applied to the training fold, and the model is tested on every house in the test fold. This matches deployment, where the model must price any house it receives.`, { indent: { left: 360 } }),
  P(`We report R² on the log scale (comparable with the original pipeline), the root mean squared error (RMSE) and mean absolute error (MAE) in US dollars, the MAPE, and the share of predictions that are off by more than 50%. Unregularised linear regression on one-hot features occasionally produced astronomically large log predictions; these were capped at |log price| = 25 so that dollar metrics remain finite, and the number of capped predictions is reported.`),
  P(`To compare two models we use the corrected resampled t-test of Nadeau and Bengio ${c("nadeau")}:`, { after: 0 }),
  equation(),
  P(`where *μ*_{d} and *s*_{d}² are the mean and variance of the per-split differences in a metric between the two models, J = 15 is the number of splits and n₂/n₁ = 1/4 is the ratio of test to training set size. The statistic has J − 1 degrees of freedom, and we use a two-sided significance level of 0.05.`),
  H2("D. Models and Features"),
  P(`We compare the five models in Table III, each with two feature sets: *numeric only* (the 36 numeric predictors, identifiers removed) and *numeric + categorical* (adding the ${ds.categorical_columns} categorical columns, one-hot encoded, with missing categories treated as their own level "None"; 319 features in total). Numeric features are mean-imputed and, for the linear models, standardised. Hyperparameters were fixed in advance and not tuned per fold, except for the internal cross-validation of ridge and lasso.`),
  table("Models and hyperparameters", ["Model", "Settings"], [
    ["Linear regression", "Ordinary least squares"],
    ["Ridge " + c("ridge"), "Regularisation strength chosen by efficient leave-one-out CV from 30 values in [10⁻², 10³]"],
    ["Lasso " + c("lasso"), "Regularisation strength chosen by inner 5-fold CV from 40 values"],
    ["Random forest " + c("rf"), "200 trees, one third of features considered per split"],
    ["XGBoost " + c("xgb"), "1,500 trees, learning rate 0.03, depth 4, row subsample 0.8, column subsample 0.5, min. child weight 2"],
  ], [0.25, 0.75], { align: [AlignmentType.LEFT, AlignmentType.LEFT] }),
  H2("E. Experiments"),
  P(`**E0** reproduces the original result on its single 80/20 split. **E1** compares five outlier strategies (none, the original global z-score, z-score on log price only, interquartile range on log price, and De Cock's rule of removing houses over 4,000 sq ft) under both protocols, using the original linear model. **E2** is an ablation that changes the original pipeline one step at a time into a corrected one. **E3** compares the models of Table III. **E4** measures permutation feature importance ${c("rf", "fisher")} for the best model on a held-out 20% split. **E5** measures pipeline run times, run-to-run reproducibility and the latency of the deployed model server.`),
  H2("F. Implementation"),
  P(`The experiments use Python 3.11.9, scikit-learn 1.3.2, XGBoost 2.1.4, pandas 2.0.3, NumPy 1.24.4, ZenML 0.64.0 and MLflow 2.15.1, on a laptop with an Intel Core i5-1145G7 processor (8 logical cores) and 16 GB of RAM running Windows 11 Pro. E0–E4 take about ${Math.round(ds.runtime_seconds / 60)} minutes in total.`, { after: 200 }),
);

// V. Results
add(H1("V. RESULTS"),
  H2("A. Reproducing the Original Result (E0)"),
  P(`Our reimplementation reproduces the original pipeline's output exactly: on its single 80/20 split it keeps ${E0.original.rows_kept.toLocaleString("en-US")} houses and obtains R² = ${E0.original.R2_log.toFixed(4)} and MSE = ${E0.original.MSE_log.toFixed(4)} on the log scale, identical to the values logged by the original pipeline. Changing only the outlier rule to a z-score on log price keeps ${E0.target_only_fix.rows_kept.toLocaleString("en-US")} houses and gives R² = ${E0.target_only_fix.R2_log.toFixed(4)}, which again matches the corrected pipeline. This confirms that the experiments below test the same model as the original system.`),
  H2("B. Outlier Strategy and Evaluation Protocol (E1)"),
  P(`Table IV and Fig. 2 show the results. Under protocol A the original global rule appears to be the best choice, with the highest R² (${pm(gA.R2_log_mean, gA.R2_log_std)}) and the lowest RMSE (${usd(gA.RMSE_usd_mean)}). Under protocol B the same rule is by far the worst: R² falls to ${pm(g.R2_log_mean, g.R2_log_std, 2)}, the RMSE rises to ${usd(g.RMSE_usd_mean)} and ${g.Pct_off_by_50pct_mean.toFixed(1)}% of predictions are off by more than 50%. The other rules are almost unaffected by the protocol and perform like no removal at all (R² ≈ 0.87). Even De Cock's well-motivated rule looks better under protocol A (R² ${num(dA.R2_log_mean)}, RMSE ${usd(dA.RMSE_usd_mean)}) than under protocol B (R² ${num(dB.R2_log_mean)}, RMSE ${usd(dB.RMSE_usd_mean)}), because it removes the five largest and hardest-to-price houses from the test set.`),
  table("Effect of outlier strategy and evaluation protocol on the original linear model (mean ± SD over 15 splits)",
    ["Outlier rule", "Rows kept", "Protocol", "R² (log)", "RMSE (US$)", "MAPE (%)", ">50% error (%)"],
    e1.map((r) => [r.rule === "None" ? "No removal" : r.rule, r.rows_kept.toLocaleString("en-US"),
      r.protocol === "Filter before split" ? "A" : "B", pm(r.R2_log_mean, r.R2_log_std), usd(r.RMSE_usd_mean),
      r.MAPE_pct_mean.toFixed(2), r.Pct_off_by_50pct_mean.toFixed(2)]),
    [0.23, 0.1, 0.09, 0.17, 0.14, 0.12, 0.15],
    { note: "Protocol A filters the whole dataset before splitting; protocol B filters only the training folds and tests on every house." }),
  ...figure("fig2_outlier_protocols.png", 6.2, "R² of the original linear model for each outlier rule under the two protocols. The original global rule is best when the test set is filtered (blue) but fails when every house is tested (orange)."),
  P(`Table V explains the failure. Under protocol B we split each test fold into the ${nRare} houses that the global rule would flag and the remaining houses. With the global rule, the model is accurate on ordinary houses (R² ${num(go.R2_log_mean)}, MAPE ${go.MAPE_pct_mean.toFixed(1)}%) but fails on houses with a rare feature (R² ${num(gr.R2_log_mean, 2)}, MAPE ${gr.MAPE_pct_mean.toFixed(1)}%, ${gr.Pct_off_by_50pct_mean.toFixed(1)}% of predictions off by more than 50%). In the filtered data these features are constant or almost constant: Kitchen AbvGr is 1 for every retained house, and Bsmt Half Bath is 0 except for a single mean-imputed missing value (0.061). Their coefficients are therefore estimated from almost no variation and can take extreme values. In the deployed original model, a single basement half-bath lowered the predicted log price by 4.91, i.e. multiplied the price by about 0.007, which produced the $1,375 valuation mentioned in Section I. With the target-only rule, the same houses are predicted with MAPE ${tr.MAPE_pct_mean.toFixed(1)}%.`),
  table("Error on houses with and without a rare feature (protocol B)",
    ["Outlier rule", "Houses", "R² (log)", "RMSE (US$)", "MAPE (%)", ">50% error (%)"],
    e1r.map((r) => [r.rule === "None" ? "No removal" : r.rule, r.group === "Other houses" ? "Other" : "Rare feature",
      num(r.R2_log_mean), usd(r.RMSE_usd_mean), r.MAPE_pct_mean.toFixed(2), r.Pct_off_by_50pct_mean.toFixed(2)]),
    [0.27, 0.15, 0.13, 0.16, 0.13, 0.16],
    { note: `"Rare feature" = one of the ${nRare} houses that the original global rule removes.` }),
  H2("C. Ablation from the Original to a Corrected Pipeline (E2)"),
  P(`Table VI changes the original pipeline one step at a time, evaluating each step under protocol B. Moving outlier detection to the target alone (step 2) accounts for almost the whole recovery, from R² ${num(L("L1").R2_log_mean, 2)} to ${num(L("L2").R2_log_mean)}. Fitting imputation inside the training folds (step 3) and removing the identifier columns (step 4) are correct practice but change the results only in the fourth decimal place on this dataset. Adding one-hot categorical features to unregularised linear regression (step 5) makes the model numerically unstable: with 319 features, some of them rare categories, the least-squares problem is ill-conditioned, and on average ${L("L5").N_capped_mean.toFixed(0)} test predictions per split hit the cap. Adding ridge regularisation (step 6) resolves this and gives the best linear result, R² = ${pm(L("L6").R2_log_mean, L("L6").R2_log_std)}, RMSE ${usd(L("L6").RMSE_usd_mean)} and MAPE ${L("L6").MAPE_pct_mean.toFixed(2)}%.`),
  table("Ablation from the original pipeline to a corrected pipeline",
    ["Step", "Change", "R² (log)", "RMSE (US$)", "MAPE (%)"],
    [["0", "Original, as reported (single split, test set filtered)", E0.original.R2_log.toFixed(3), "n/a", "n/a"],
     ...e2.map((r) => [r.step.slice(1, 2), r.step.slice(4).replace(/^\+ /, "+ "), pm(r.R2_log_mean, r.R2_log_std), usd(r.RMSE_usd_mean), num(r.MAPE_pct_mean, 2)])],
    [0.07, 0.43, 0.17, 0.17, 0.16],
    { align: [AlignmentType.CENTER, AlignmentType.LEFT, AlignmentType.CENTER, AlignmentType.CENTER, AlignmentType.CENTER], note: "Steps 1–6 use protocol B with 15 splits (mean ± SD); each step keeps all previous changes." }),
  H2("D. Model Comparison (E3)"),
  P(`Table VII and Fig. 3 compare the models under the corrected protocol. XGBoost with numeric and categorical features is best on every metric: R² = ${pm(best.R2_log_mean, best.R2_log_std)}, RMSE ${usd(best.RMSE_usd_mean)}, MAE ${usd(best.MAE_usd_mean)} and MAPE ${best.MAPE_pct_mean.toFixed(2)}%. Categorical features improve every model except unregularised linear regression; for ridge, R² rises from ${num(m3("Ridge", NO).R2_log_mean)} to ${num(m3("Ridge", NC).R2_log_mean)}, and for XGBoost from ${num(m3("XGBoost", NO).R2_log_mean)} to ${num(m3("XGBoost", NC).R2_log_mean)}. The tree ensembles also have far more stable errors across splits (RMSE standard deviation about $3,000) than the linear models ($13,000–$24,000), because linear models extrapolate badly on a few very large or unusual houses, and those houses dominate the RMSE of the splits that contain them.`),
  table("Model comparison under the corrected protocol (mean ± SD over 15 splits)",
    ["Model", "Features", "R² (log)", "RMSE (US$)", "MAE (US$)", "MAPE (%)", "Fit time (s)"],
    e3.map((r) => [r.model, r.features === NC ? "Num. + cat." : "Numeric", pm(r.R2_log_mean, r.R2_log_std),
      usd(r.RMSE_usd_mean), usd(r.MAE_usd_mean), num(r.MAPE_pct_mean, 2), r.fit_seconds_per_fold.toFixed(2)]),
    [0.18, 0.13, 0.18, 0.15, 0.13, 0.11, 0.12],
    { note: "Fit time is per split on the hardware in Section IV-F. Linear regression with categorical features was numerically unstable (Section V-C)." }),
  ...figure("fig3_model_comparison.png", 6.2, "Mean RMSE in US dollars of each model with numeric features only (blue) and with numeric and categorical features (orange). Lower is better."),
  P(`Table VIII tests whether the best model's advantage is statistically significant. XGBoost with categorical features is significantly better than every numeric-only model on log RMSE and MAE (p ≤ ${Math.max(s3("Ridge", NO).RMSE_log_p, s3("Linear Regression", NO).RMSE_log_p, s3("Lasso", NO).RMSE_log_p).toFixed(3)}) and than both random forests on all three metrics (p < 0.001). Its advantage over ridge and lasso with categorical features is not significant (log RMSE p = ${s3("Ridge", NC).RMSE_log_p.toFixed(2)} and ${s3("Lasso", NC).RMSE_log_p.toFixed(2)}; MAE p = ${s3("Ridge", NC).MAE_usd_p.toFixed(3)} and ${s3("Lasso", NC).MAE_usd_p.toFixed(3)}). A regularised linear model with categorical features is therefore a strong, fast and interpretable alternative: it fits in about one second per split, compared with ${best.fit_seconds_per_fold.toFixed(1)} s for XGBoost.`),
  table("Corrected resampled t-test: XGBoost (num. + cat.) against each other configuration",
    ["Model", "Features", "Δ RMSE (log)", "p", "Δ MAE (US$)", "p", "Δ RMSE (US$)", "p"],
    sig.map((r) => [r.model, r.features === NC ? "Num. + cat." : "Numeric", num(r.RMSE_log_diff, 4), pval(r.RMSE_log_p),
      usd(r.MAE_usd_diff), pval(r.MAE_usd_p), usd(r.RMSE_usd_diff), pval(r.RMSE_usd_p)]),
    [0.17, 0.12, 0.12, 0.09, 0.14, 0.09, 0.17, 0.10],
    { note: "Δ = other model minus XGBoost (positive means XGBoost has the lower error). Two-sided p-values, 14 degrees of freedom." }),
  H2("E. Prediction Quality and Feature Importance (E4)"),
  P(`On a held-out 20% split (586 houses), the best model reaches R² = ${ho.R2_log.toFixed(3)} on the log scale and ${ho.R2_usd.toFixed(3)} in dollars, with RMSE ${usd(ho.RMSE_usd)}, MAE ${usd(ho.MAE_usd)} and MAPE ${ho.MAPE_pct.toFixed(2)}%. Fig. 4 plots cross-validated predictions against actual prices. Predictions follow the diagonal closely up to about $400,000; above that the model tends to under-predict, since only 17 houses in the dataset sold for more than $500,000.`),
  ...figure("fig4_pred_vs_actual.png", 3.6, "Cross-validated predictions of XGBoost (numeric + categorical) against actual sale prices; each house is predicted once by a model that did not see it during training."),
  P(`Fig. 5 shows permutation importance, i.e. how much the log RMSE increases when one feature's values are shuffled. Above-ground living area (${imp[0].importance_mean.toFixed(3)}) and overall quality (${imp[1].importance_mean.toFixed(3)}) dominate, followed by year built, overall condition and basement area. Neighbourhood is the most important categorical feature. This ranking agrees with hedonic theory, where size, quality, age and location drive price ${c("rosen")}. Correlated features, such as living area and first-floor area, share importance, so individual values should be read with care ${c("fisher")}.`),
  ...figure("fig5_feature_importance.png", 5.2, "The ten most important features of the best model by permutation importance on the held-out split (mean ± SD over 10 shuffles)."),
  H2("F. MLOps Behaviour (E5)"),
  P(`Table IX summarises the MLOps measurements. Training runs that used the same code produced identical evaluation scores to all 16 printed decimal places: the ${mlops.distinct_R2_values["0.9221139768594216"]} runs of the original pipeline gave R² = 0.9221139768594216, and the ${mlops.distinct_R2_values["0.8567861675345597"]} runs after the outlier fix gave 0.8567861675345597. The deployed model server answers a single-house request in a median of ${mlops.latency_single.median_ms.toFixed(1)} ms, and a 100-house batch in ${mlops.latency_batch100.median_ms.toFixed(1)} ms. Latency is therefore dominated by HTTP and framework overhead rather than by the model, and batching reduces the cost per house to under 1 ms. These latency figures are for the linear model that the ZenML pipeline deploys. Fig. 6 shows the MLflow dashboard used to track the runs.`),
  table("MLOps measurements on the Windows laptop", ["Measurement", "Value"], [
    ["Completed training pipeline runs", String(mlops.training_runs_completed)],
    ["Training pipeline duration: median (min–max)", `${mlops.training_duration_s.median.toFixed(1)} s (${mlops.training_duration_s.min.toFixed(1)}–${mlops.training_duration_s.max.toFixed(1)} s)`],
    ["Distinct R² values across runs with the same code", "1 per code version (identical to 16 decimal places)"],
    ["Inference pipeline duration (median of " + mlops.inference_runs_completed + " runs)", `${mlops.inference_duration_s_median.toFixed(1)} s`],
    ["Model server cold start", `${mlops.server_startup_s.toFixed(1)} s`],
    ["Latency, 1 house per request (200 requests): median / p95", `${mlops.latency_single.median_ms.toFixed(1)} / ${mlops.latency_single.p95_ms.toFixed(1)} ms`],
    ["Latency, 100 houses per request (50 requests): median / p95", `${mlops.latency_batch100.median_ms.toFixed(1)} / ${mlops.latency_batch100.p95_ms.toFixed(1)} ms`],
  ], [0.62, 0.38]),
  ...figurePlaceholder("MLflow tracking dashboard showing the training runs of the ml_pipeline experiment.",
    "Insert a screenshot of the MLflow dashboard here: double-click 4_open_mlflow_dashboard.bat, open the ml_pipeline experiment and capture the list of runs"),
);

// VI. Discussion
add(H1("VI. DISCUSSION"),
  H2("A. Why Preprocessing Inflated the Accuracy"),
  P(`The original pipeline's R² of 0.922 was not wrong arithmetic; it described a different population from the one the model would serve. Removing rows before the split changes the test set, so the metric no longer measures performance on the houses the model will be asked to price. The z-score rule made this worse because it assumes roughly normal data, while many housing features are zero for most houses; on such features every non-zero value is an "outlier". The result was a model that looked best in evaluation and failed on almost a third of real houses. Kapoor and Narayanan list preprocessing on training and test data together as a common leakage type ${c("kapoor")}; our results show that it can change not only the size of a reported metric but also which modelling choice looks best.`),
  P(`From these results we suggest five practical checks for similar projects:`, { after: 60 }),
  NUM(`Fit every data-dependent step, including outlier thresholds, on training data only, ideally inside a single pipeline object.`, "checks"),
  NUM(`Never filter the test set unless the same filter will run in production and the excluded cases are handled separately.`, "checks"),
  NUM(`Choose outlier rules per variable with domain reasoning, and log how many rows each step removes.`, "checks"),
  NUM(`Report errors in the original units and for subgroups, not only R² on a transformed scale.`, "checks"),
  NUM(`Inspect individual predictions for plausibility before trusting an aggregate metric.`, "checks"),
  H2("B. Choosing a Model"),
  P(`Under a fair protocol, categorical information and regularisation matter more than the choice of algorithm family. XGBoost gave the lowest errors and the most stable results across splits, but ridge and lasso with categorical features were not significantly worse on log RMSE, and they train about eight times faster and are easier to explain. The right choice depends on whether accuracy, speed or interpretability matters most.`),
  H2("C. Lessons for MLOps"),
  P(`The MLOps tooling made the problem easy to diagnose: versioned artefacts and tracked runs allowed an exact reproduction of the original score, and repeated runs were deterministic. However, a pipeline automates whatever evaluation it encodes, and here it reproduced a misleading score reliably. Automated checks on data volume, such as an alert when a cleaning step removes more than a few percent of rows, would have caught the problem early. Portability is a second practical issue: a deployment component that silently assumes Unix daemons prevents the pipeline from running on Windows machines, which are common in teaching and enterprise environments.`, { after: 200 }),
);

// VII. Limitations
add(H1("VII. LIMITATIONS AND THREATS TO VALIDITY"),
  P(`This study uses a single dataset from one city and the period 2006–2010, which includes the 2008 financial crisis; results may differ for other markets. Random cross-validation ignores time, so a model trained on later sales and tested on earlier ones is included; a time-based split would be more realistic. Hyperparameters were fixed rather than tuned, which may disadvantage the tree ensembles. Cross-validation splits overlap; the corrected t-test reduces but does not remove this dependence, and the nine comparisons in Table VIII were not corrected for multiple testing. Permutation importance can be misleading for correlated features. The MLOps timings come from one laptop and will vary with hardware. Finally, predictions of the unstable linear model were capped to keep dollar metrics finite.`, { after: 200 }),
);

// VIII. Conclusion
add(H1("VIII. CONCLUSION AND FUTURE WORK"),
  P(`We examined an open-source MLOps pipeline for house price prediction and showed that its reported R² of ${E0.original.R2_log.toFixed(3)} was produced by an outlier step that removed ${pctRare}% of houses from both training and test data. Evaluated on all houses, the same model scored R² = ${num(g.R2_log_mean, 2)}; limiting outlier detection to the target restored R² = ${num(tB.R2_log_mean)}. Under a leakage-aware protocol, XGBoost with numeric and categorical features performed best (R² = ${num(best.R2_log_mean)}, MAPE = ${best.MAPE_pct_mean.toFixed(1)}%), with regularised linear models statistically close behind. We also made the pipeline deployable on Windows and showed that its runs are exactly reproducible.`),
  P(`Future work includes time-based validation, hyperparameter tuning as a pipeline step, deploying the best model through the pipeline, prediction intervals that quantify uncertainty, and automated data checks and drift monitoring in production.`, { after: 200 }),
);

// Acknowledgment
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 }, children: [new TextRun({ text: "ACKNOWLEDGMENT", font: FONT, size: 20 })] }),
  P(`The author thanks [[Guide's name]] for guidance. The baseline pipeline was adapted from the open-source project in ${c("orig")}; Section IV-A lists the changes made to it. [[If your institution requires it, state here how AI tools were used in preparing this work.]]`, { after: 200 }),
);

// References
const refParas = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 120 }, children: [new TextRun({ text: "REFERENCES", font: FONT, size: 20 })] })];
order.forEach((k, i) => refParas.push(new Paragraph({
  alignment: AlignmentType.LEFT, spacing: { after: 60 },
  indent: { left: 480, hanging: 480 },
  tabStops: [{ type: TabStopType.LEFT, position: 480 }],
  children: [new TextRun({ text: `[${i + 1}]\t`, font: FONT, size: 18 }), ...runs(REFS[k], { size: 18 })],
})));
const unused = Object.keys(REFS).filter((k) => !order.includes(k));
if (unused.length) throw new Error("Uncited references: " + unused.join(", "));
add(refParas);

// Appendix
add(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 }, children: [new TextRun({ text: "APPENDIX: REPRODUCING THE RESULTS", font: FONT, size: 20 })] }),
  P(`All results can be regenerated on Windows from the project folder. Run setup_windows.bat once, then:`, { after: 60 }),
  ...[
    `venv\\Scripts\\python.exe research\\experiments.py  (E0–E4, about ${Math.round(summary.runtime_seconds / 60)} minutes)`,
    "venv\\Scripts\\python.exe research\\significance_all_metrics.py  (Table VIII)",
    "venv\\Scripts\\python.exe research\\mlops_measurements.py  (E5, Table IX)",
    "venv\\Scripts\\python.exe research\\make_figures.py  (Figs. 1–5)",
    "venv\\Scripts\\python.exe run_deployment.py  (train, deploy and run batch inference)",
  ].map((t) => new Paragraph({ spacing: { after: 40 }, indent: { left: 360 }, children: [new TextRun({ text: t, font: "Consolas", size: 17 })] })),
  P(`Results are written to research/results and figures to research/figures.`, { after: 0 }),
);

// ---------------------------------------------------------------- document
const doc = new Document({
  creator: "Term paper",
  title: "When Data Cleaning Inflates Accuracy",
  styles: {
    default: { document: { run: { font: FONT, size: 21 } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 21, color: "000000" },
        paragraph: { alignment: AlignmentType.CENTER, spacing: { before: 240, after: 120 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 21, italics: true, color: "000000" },
        paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 1 } },
    ],
  },
  numbering: {
    config: ["contrib", "checks"].map((reference) => ({
      reference,
      levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1)", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 720, hanging: 360 } } } }],
    })),
  },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER,
      children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 18 })] })] }) },
    children: body,
  }],
});

Packer.toBuffer(doc).then((buf) => {
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  fs.writeFileSync(OUT, buf);
  console.log("wrote", OUT, `(${order.length} references, ${tabNo} tables, ${figNo} figures)`);
});
