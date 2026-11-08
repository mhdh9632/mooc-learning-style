#!/usr/bin/env node

const fs = require('fs');
const path = require('path');

const root = path.resolve(__dirname, '..');

const activeNotebooks = [
  'models/baseline_bilstm/01_event_extraction_and_scaling.ipynb',
  'models/baseline_bilstm/02_sequence_preprocessing.ipynb',
  'models/baseline_bilstm/03_bilstm_training_and_evaluation.ipynb',
  'models/fslsm_bilstm/01_fslsm_sequence_preprocessing.ipynb',
  'models/fslsm_bilstm/02_fslsm_bilstm_training_and_evaluation.ipynb',
  'models/fslsm_bilstm/03_all_dimensions_event_level_bilstm.ipynb',
  'models/transformer/fslsm_transformer.ipynb',
  'models/sequence_representation/00_shared_setup.ipynb',
  'models/sequence_representation/01_markov_mixture.ipynb',
  'models/sequence_representation/02_next_token_gru.ipynb',
  'models/sequence_representation/03_gru_autoencoder.ipynb',
  'models/sequence_representation/04_contrastive_gru.ipynb',
  'models/sequence_representation/05_sequential_gnn.ipynb',
  'models/sequence_representation/06_model_comparison.ipynb',
  'models/feature_engineering/event_scaling_and_kmeans.ipynb',
  'models/feature_engineering/fslsm_all_dimensions_scoring.ipynb',
  'models/feature_engineering/proportional_features.ipynb',
];

const errors = [];
const notebookTexts = new Map();

const requiredGuides = [
  'PROJECT.md',
  'models/baseline_bilstm/README.md',
  'models/fslsm_bilstm/README.md',
  'models/transformer/README.md',
  'models/sequence_representation/README.md',
  'models/feature_engineering/README.md',
  'archive/README.md',
];

for (const relativePath of requiredGuides) {
  if (!fs.existsSync(path.join(root, relativePath))) {
    errors.push(`Missing required guide: ${relativePath}`);
  }
}

for (const relativePath of activeNotebooks) {
  const absolutePath = path.join(root, relativePath);
  if (!fs.existsSync(absolutePath)) {
    errors.push(`Missing active notebook: ${relativePath}`);
    continue;
  }

  const raw = fs.readFileSync(absolutePath, 'utf8');
  try {
    const notebook = JSON.parse(raw);
    const text = (notebook.cells || [])
      .map((cell) => (cell.source || []).join(''))
      .join('\n');
    notebookTexts.set(relativePath, text);
  } catch (error) {
    errors.push(`Invalid notebook JSON: ${relativePath} (${error.message})`);
  }
}

for (const [relativePath, text] of notebookTexts) {
  if (text.includes('data/raw/')) {
    errors.push(`Unresolved local-only data/raw path: ${relativePath}`);
  }

  if (/KAGGLE_KEY\s*=\s*["'][^"']+["']/.test(text)) {
    errors.push(`Possible embedded Kaggle secret: ${relativePath}`);
  }

  if (/"key"\s*:\s*"[A-Za-z0-9_-]{20,}"/.test(text)) {
    errors.push(`Possible embedded credential JSON: ${relativePath}`);
  }

  if (text.includes('torch.load(') && !/torch\.load\([^)]*weights_only\s*=\s*False/.test(text)) {
    errors.push(`torch.load must explicitly allow the trusted mixed-data artifact: ${relativePath}`);
  }
}

function walkMarkdown(directory) {
  if (!fs.existsSync(directory)) return [];
  return fs.readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const absolutePath = path.join(directory, entry.name);
    if (entry.isDirectory()) return walkMarkdown(absolutePath);
    return entry.name.endsWith('.md') ? [absolutePath] : [];
  });
}

for (const markdownPath of [path.join(root, 'PROJECT.md'), ...walkMarkdown(path.join(root, 'models'))]) {
  if (!fs.existsSync(markdownPath)) continue;
  const markdown = fs.readFileSync(markdownPath, 'utf8');
  const linkPattern = /\[[^\]]*\]\(([^)]+)\)/g;
  for (const match of markdown.matchAll(linkPattern)) {
    const target = match[1].trim().replace(/^<|>$/g, '').split('#')[0];
    if (!target || /^(https?:|mailto:)/.test(target)) continue;
    const resolved = path.resolve(path.dirname(markdownPath), target);
    if (!fs.existsSync(resolved)) {
      errors.push(`Broken documentation link in ${path.relative(root, markdownPath)}: ${match[1]}`);
    }
  }
}

if (errors.length > 0) {
  console.error(`Project validation failed with ${errors.length} issue(s):`);
  for (const error of errors) console.error(`- ${error}`);
  process.exit(1);
}

console.log(`Project validation passed: ${activeNotebooks.length} active notebooks are valid JSON with safe, portable paths.`);
