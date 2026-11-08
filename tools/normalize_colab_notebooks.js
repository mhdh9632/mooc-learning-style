#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

function sourceLines(text) {
  return text.split(/(?<=\n)/);
}

function setupCells(modelArtifactName, extra = '') {
  const markdown = {
    cell_type: 'markdown',
    metadata: { tags: ['colab-config'] },
    source: sourceLines(
      '## Google Colab and data paths\n\n' +
      'Set `KDD_PROJECT_ROOT` if your project is not stored at ' +
      '`/content/drive/MyDrive/kdd_dropout_project`. The configured folder must contain ' +
      '`dataset/`, `shared/metadata/`, and `artifacts/`.\n'
    ),
  };

  const code = {
    cell_type: 'code',
    execution_count: null,
    metadata: { tags: ['colab-config'] },
    outputs: [],
    source: sourceLines(
      'from pathlib import Path\n' +
      'import os\n\n' +
      'IN_COLAB = False\n' +
      'try:\n' +
      '    from google.colab import drive\n' +
      '    drive.mount("/content/drive")\n' +
      '    IN_COLAB = True\n' +
      'except ImportError:\n' +
      '    pass\n\n' +
      'def discover_project_root(start):\n' +
      '    start = Path(start).resolve()\n' +
      '    for candidate in (start, *start.parents):\n' +
      '        if (candidate / "dataset").exists() and (candidate / "shared" / "metadata").exists():\n' +
      '            return candidate\n' +
      '    return start\n\n' +
      'default_root = (\n' +
      '    Path("/content/drive/MyDrive/kdd_dropout_project")\n' +
      '    if IN_COLAB else discover_project_root(Path.cwd())\n' +
      ')\n' +
      'PROJECT_ROOT = Path(os.environ.get("KDD_PROJECT_ROOT", default_root))\n' +
      'DATA_ROOT = PROJECT_ROOT / "dataset"\n' +
      'METADATA_ROOT = PROJECT_ROOT / "shared" / "metadata"\n' +
      'ARTIFACT_ROOT = PROJECT_ROOT / "artifacts"\n' +
      `MODEL_ARTIFACT_ROOT = ARTIFACT_ROOT / "${modelArtifactName}"\n` +
      'MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)\n' +
      extra +
      'print(f"Project root: {PROJECT_ROOT}")\n' +
      'print(f"Model artifacts: {MODEL_ARTIFACT_ROOT}")\n'
    ),
  };

  return [markdown, code];
}

function replaceInCode(notebook, replacements) {
  for (const cell of notebook.cells || []) {
    if (cell.cell_type !== 'code') continue;
    let code = (cell.source || []).join('');
    for (const [before, after] of replacements) {
      code = code.split(before).join(after);
    }
    cell.source = sourceLines(code);
  }
}

function updateNotebook(relativePath, modelArtifactName, replacements, extra = '') {
  const absolutePath = path.join(root, relativePath);
  const notebook = JSON.parse(fs.readFileSync(absolutePath, 'utf8'));
  const alreadyConfigured = (notebook.cells || []).some(
    (cell) => (cell.metadata?.tags || []).includes('colab-config')
  );
  if (!alreadyConfigured) notebook.cells.unshift(...setupCells(modelArtifactName, extra));
  replaceInCode(notebook, replacements);
  fs.writeFileSync(absolutePath, `${JSON.stringify(notebook, null, 1)}\n`);
}

const commonRawReplacements = [
  ['pd.read_csv("data/raw/train/train/log_train.csv")', 'pd.read_csv(DATA_ROOT / "train/train/log_train.csv")'],
  ['pd.read_csv("data/raw/test/test/log_test.csv")', 'pd.read_csv(DATA_ROOT / "test/test/log_test.csv")'],
  ['pd.read_csv("data/raw/train/train/truth_train.csv"', 'pd.read_csv(DATA_ROOT / "train/train/truth_train.csv"'],
  ['pd.read_csv("data/raw/test/test/truth_test.csv"', 'pd.read_csv(DATA_ROOT / "test/test/truth_test.csv"'],
  ['pd.read_csv("data/raw/object.csv")', 'pd.read_csv(METADATA_ROOT / "object.csv")'],
];

updateNotebook(
  'models/baseline_bilstm/01_event_extraction_and_scaling.ipynb',
  'baseline_bilstm',
  [
    ...commonRawReplacements,
    ['os.makedirs("data/temporal_event_counts", exist_ok=True)', 'TEMPORAL_COUNTS_ROOT.mkdir(parents=True, exist_ok=True)'],
    ['out_path = f"data/temporal_event_counts/{name}_norm.csv"', 'out_path = TEMPORAL_COUNTS_ROOT / f"{name}_norm.csv"'],
    ['norm_path = f"data/temporal_event_counts/{slice_name}_norm.csv"', 'norm_path = TEMPORAL_COUNTS_ROOT / f"{slice_name}_norm.csv"'],
  ],
  'TEMPORAL_COUNTS_ROOT = MODEL_ARTIFACT_ROOT / "temporal_event_counts"\nTEMPORAL_COUNTS_ROOT.mkdir(parents=True, exist_ok=True)\n'
);

updateNotebook(
  'models/baseline_bilstm/02_sequence_preprocessing.ipynb',
  'baseline_bilstm',
  [
    ...commonRawReplacements,
    ['os.makedirs("data/processed", exist_ok=True)', 'MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)'],
    ['"data/processed/processed_data.pt"', 'MODEL_ARTIFACT_ROOT / "processed_data.pt"'],
    ['print("Success! Cleaned data saved to data/processed/processed_data.pt")', 'print(f"Success! Cleaned data saved to {MODEL_ARTIFACT_ROOT / \'processed_data.pt\'}")'],
  ]
);

updateNotebook(
  'models/baseline_bilstm/03_bilstm_training_and_evaluation.ipynb',
  'baseline_bilstm',
  [
    ['"data/processed/processed_data.pt"', 'MODEL_ARTIFACT_ROOT / "processed_data.pt"'],
    ['torch.load(MODEL_ARTIFACT_ROOT / "processed_data.pt")', 'torch.load(MODEL_ARTIFACT_ROOT / "processed_data.pt", weights_only=False)  # Trust only the artifact created by notebook 02'],
  ]
);

updateNotebook(
  'models/fslsm_bilstm/01_fslsm_sequence_preprocessing.ipynb',
  'fslsm_bilstm',
  [
    ...commonRawReplacements,
    ['os.makedirs("data/processed", exist_ok=True)', 'MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)'],
    ['"data/processed/processed_data_fslsm.pt"', 'MODEL_ARTIFACT_ROOT / "processed_data_fslsm.pt"'],
    ['print("Success! Saved to data/processed/processed_data_fslsm.pt")', 'print(f"Success! Saved to {MODEL_ARTIFACT_ROOT / \'processed_data_fslsm.pt\'}")'],
  ]
);

updateNotebook(
  'models/fslsm_bilstm/02_fslsm_bilstm_training_and_evaluation.ipynb',
  'fslsm_bilstm',
  [
    ['"data/processed/processed_data_fslsm.pt"', 'MODEL_ARTIFACT_ROOT / "processed_data_fslsm.pt"'],
    ['torch.load(MODEL_ARTIFACT_ROOT / "processed_data_fslsm.pt")', 'torch.load(MODEL_ARTIFACT_ROOT / "processed_data_fslsm.pt", weights_only=False)  # Trust only the artifact created by notebook 01'],
  ]
);

updateNotebook(
  'models/feature_engineering/fslsm_all_dimensions_scoring.ipynb',
  'feature_engineering',
  [
    ...commonRawReplacements,
    ['os.makedirs("data/processed", exist_ok=True)', 'MODEL_ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)'],
    ['"data/processed/fslsm_full_train.csv"', 'MODEL_ARTIFACT_ROOT / "fslsm_full_train.csv"'],
    ['"data/processed/fslsm_full_test.csv"', 'MODEL_ARTIFACT_ROOT / "fslsm_full_test.csv"'],
    ['print("\\nSaved: data/processed/fslsm_full_train.csv, data/processed/fslsm_full_test.csv")', 'print(f"\\nSaved: {MODEL_ARTIFACT_ROOT / \'fslsm_full_train.csv\'}, {MODEL_ARTIFACT_ROOT / \'fslsm_full_test.csv\'}")'],
  ]
);

updateNotebook(
  'models/feature_engineering/proportional_features.ipynb',
  'feature_engineering',
  [
    ...commonRawReplacements,
    ['RAW_OUTPUT_DIR = "data/temporal_event_counts"', 'RAW_OUTPUT_DIR = MODEL_ARTIFACT_ROOT / "temporal_event_counts"'],
    ['PROP_OUTPUT_DIR = "data/temporal_event_proportions"', 'PROP_OUTPUT_DIR = MODEL_ARTIFACT_ROOT / "temporal_event_proportions"'],
    ['FEATURE_DIR = "data/temporal_event_counts"', 'FEATURE_DIR = MODEL_ARTIFACT_ROOT / "temporal_event_counts"'],
    ['FEATURE_DIR = "data/temporal_event_proportions"', 'FEATURE_DIR = MODEL_ARTIFACT_ROOT / "temporal_event_proportions"'],
    ['norm_path = f"{FEATURE_DIR}/{slice_name}_norm.csv"', 'norm_path = FEATURE_DIR / f"{slice_name}_norm.csv"'],
  ]
);

const sequenceSetupPath = path.join(root, 'models/sequence_representation/00_shared_setup.ipynb');
const sequenceSetup = JSON.parse(fs.readFileSync(sequenceSetupPath, 'utf8'));
replaceInCode(sequenceSetup, [
  [
    'DATA_ROOT = "/content/drive/MyDrive/kdd2015"   # folder that contains train/, test/, object.csv\nWORK      = "/content/drive/MyDrive/kdd_seq"',
    'DATA_ROOT  = "/content/drive/MyDrive/kdd2015"  # folder that contains train/ and test/\nOBJECT_PATH = f"{DATA_ROOT}/object.csv"             # may point to shared/metadata/object.csv separately\nWORK       = "/content/drive/MyDrive/kdd_seq"',
  ],
  ['obj=f"{DATA_ROOT}/object.csv")', 'obj=OBJECT_PATH)'],
  ['Fix DATA_ROOT / paths, not found:', 'Fix DATA_ROOT, OBJECT_PATH, or other paths; not found:'],
]);
fs.writeFileSync(sequenceSetupPath, `${JSON.stringify(sequenceSetup, null, 1)}\n`);

console.log('Normalized Colab configuration and paths in 8 active notebooks.');
