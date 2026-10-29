'use strict';
const $ = id => document.getElementById(id);
let mode = 'upload', selectedFile = null, previewUrl = null, csrf = '', catalog = [], busy = false, indexed = 0;
const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' });
function showError(message = '') { $('error').textContent = message; $('error').hidden = !message; }
function setMode(next) {
  mode = next;
  for (const name of ['upload', 'url']) {
    const active = name === next;
    $(name + '-tab').classList.toggle('active', active);
    $(name + '-tab').setAttribute('aria-selected', active);
    $(name + '-panel').hidden = !active;
  }
  $('image-url').required = next === 'url';
  showError();
}
for (const name of ['upload', 'url']) $(name + '-tab').addEventListener('click', () => setMode(name));
document.querySelector('.tabs').addEventListener('keydown', event => {
  if (['ArrowLeft', 'ArrowRight'].includes(event.key)) { event.preventDefault(); setMode(mode === 'upload' ? 'url' : 'upload'); $(mode + '-tab').focus(); }
});
function chooseFile(file) {
  showError();
  if (!file) return;
  if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) return showError('Choose a JPG, PNG, or WebP image.');
  if (file.size > 5 * 1024 * 1024) return showError('Please choose an image smaller than 5 MiB.');
  selectedFile = file;
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('preview').src = previewUrl;
  $('preview').hidden = false; $('upload-prompt').hidden = true; $('replace-hint').hidden = false;
  $('file-name').textContent = file.name;
  setMode('upload');
}
$('file').addEventListener('change', event => chooseFile(event.target.files[0]));
for (const type of ['dragenter', 'dragover']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.add('dragging'); });
for (const type of ['dragleave', 'drop']) $('dropzone').addEventListener(type, event => { event.preventDefault(); $('dropzone').classList.remove('dragging'); });
$('dropzone').addEventListener('drop', event => chooseFile(event.dataTransfer.files[0]));
window.addEventListener('dragover', event => event.preventDefault());
window.addEventListener('drop', event => event.preventDefault());
function node(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text !== undefined) el.textContent = text; return el; }
function renderProducts(products, results = false) {
  $('product-grid').replaceChildren();
  for (const product of products) {
    const card = node('article', 'product-card');
    const picture = node('div', 'product-image');
    const img = node('img'); img.src = product.image_url; img.alt = product.title; img.loading = 'lazy';
    picture.append(img, node('span', 'category-badge', product.subcategory));
    if (results) picture.append(node('span', 'rank-badge', String(product.rank).padStart(2, '0')));
    const info = node('div', 'product-info');
    info.append(node('p', 'brand', product.brand || 'Brand not provided'), node('h3', '', product.title));
    const bottom = node('div', 'product-bottom'); bottom.append(node('span', 'price', currency.format(product.price_usd)));
    if (results) { const score = node('span', 'score', product.similarity_score.toFixed(4) + ' similarity'); score.title = 'Cosine similarity: ' + product.similarity_score; bottom.append(score); }
    info.append(bottom); card.append(picture, info); $('product-grid').append(card);
  }
  $('empty').hidden = products.length !== 0;
  $('score-note').hidden = !results;
  $('reset-results').hidden = !results;
  $('result-count').textContent = results ? products.length + ' matches' : catalog.length + ' products';
  $('results-eyebrow').textContent = results ? 'VISUAL CONNECTIONS' : 'THE COLLECTION';
  $('results-title').textContent = results ? 'Your closest matches' : 'A world of possibilities';
  $('results-description').textContent = results ? 'Ranked by cosine similarity, highest first.' : 'A few things to spark your curiosity.';
}
function collection() {
  const picks = [87, 174, 47, 6, 159, 152, 190, 12, 154];
  const products = picks.map(id => catalog.find(p => p.id === id)).filter(Boolean);
  renderProducts(products.length ? products : catalog.slice(0, 9));
}
$('reset-results').addEventListener('click', collection);
async function getJSON(response) {
  let data;
  try { data = await response.json(); } catch { throw new Error('The server returned an unreadable response. Please retry.'); }
  if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Check your image and number of matches, then retry.');
  return data;
}
async function session() { const data = await getJSON(await fetch('/ui/session')); csrf = data.csrf_token; }
function setBusy(value) {
  busy = value; $('search-button').disabled = value || !indexed;
  $('button-label').textContent = value ? 'Finding your matches…' : 'Find similar products';
  $('loading').hidden = !value; $('product-grid').hidden = value;
  $('search-form').setAttribute('aria-busy', value);
}
$('search-form').addEventListener('submit', async event => {
  event.preventDefault(); if (busy) return; showError();
  const topK = Number($('top-k').value);
  if (!Number.isInteger(topK) || topK < 1 || topK > indexed) return showError(`Choose between 1 and ${indexed} matches.`);
  if (mode === 'upload' && !selectedFile) return showError('Add a product image to begin your search.');
  const currentMode = mode;
  setBusy(true);
  try {
    await session();
    const headers = { 'X-CSRF-Token': csrf };
    let body;
    if (currentMode === 'upload') { body = new FormData(); body.append('file', selectedFile); body.append('top_k', String(topK)); }
    else { body = JSON.stringify({ image_url: $('image-url').value.trim(), top_k: topK }); headers['Content-Type'] = 'application/json'; }
    const response = await fetch(currentMode === 'upload' ? '/ui/search' : '/ui/search/url', { method: 'POST', headers, body });
    const results = await getJSON(response); renderProducts(results, true);
    if (window.innerWidth < 701) $('results-title').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) { showError(error.message || 'Search failed. Please try again.'); }
  finally { setBusy(false); }
});
async function initialize() {
  try {
    const [health, products] = await Promise.all([fetch('/health').then(getJSON), fetch('/products').then(getJSON)]);
    catalog = products; indexed = health.indexed_products;
    $('status-text').textContent = health.status === 'ok' ? 'Collection ready' : 'Index not ready';
    $('status-dot').classList.toggle('ready', health.status === 'ok');
    $('top-k').max = Math.max(indexed, 1); $('top-k').value = Math.min(5, indexed || 5);
    collection();
    for (const id of [174, 87, 6]) {
      const product = catalog.find(p => p.id === id); if (!product) continue;
      const button = node('button'); button.type = 'button'; button.title = 'Try ' + product.title; button.setAttribute('aria-label', 'Use ' + product.title + ' as your search image');
      const img = node('img'); img.src = product.image_url; img.alt = product.title; button.append(img);
      button.addEventListener('click', async () => {
        try { const response = await fetch(product.image_url); if (!response.ok) throw new Error('Could not load example image.'); const blob = await response.blob(); chooseFile(new File([blob], product.title + '.jpg', { type: blob.type })); }
        catch (error) { showError(error.message); }
      }); $('example-images').append(button);
    }
    if (health.status !== 'ok') showError(health.detail || 'Set NVIDIA_API_KEY in .env and restart the server.');
    setBusy(false);
  } catch (error) { $('status-text').textContent = 'Connection unavailable'; showError(error.message); }
}
initialize();
