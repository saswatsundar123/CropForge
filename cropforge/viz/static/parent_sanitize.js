(function () {
  const REPLACEMENTS = [
    [/\u00e2\u00ac\u2021\s*/g, ''],
    [/\u00e2\u2014\u20ac\s*/g, ''],
    [/\u00e2\u2013\u00b6/g, ''],
    [/\u00e2\u0153\u2022/g, 'Close'],
    [/\u00e2\u203a\u00b0\u00ef\u00b8\u008f\s*/g, ''],
    [/\u00e2\u20ac\u201d/g, '-'],
    [/\u00e2\u20ac\u00a6/g, '...'],
    [/\u00e2\u2020\u2019/g, '->'],
    [/\u00c3\u2014/g, 'x'],
    [/\u00c2\u00b2/g, '2'],
    [/\u00c2\u00a7/g, 'Section '],
    [/\u00c2\u00b7/g, '-'],
    [/\?{4,}/g, '-'],
  ];

  const SPECIFIC_TEXT = {
    'export-csv-btn': 'Export CSV',
    'toggle-left-btn': 'L',
    'toggle-right-btn': 'R',
    'close-terrain-modal-btn': 'Close',
    'inspector-close-btn': 'Close',
    'open-terrain-modal-btn': 'Open 3D Terrain Map',
  };

  function cleanText(value) {
    if (!value || typeof value !== 'string') return value;
    let next = value;
    for (const [pattern, replacement] of REPLACEMENTS) {
      next = next.replace(pattern, replacement);
    }
    next = next
      .replace(/LAI\s*\(m2\/m2\)/g, 'LAI (m2/m2)')
      .replace(/\s+Export CSV/g, ' Export CSV')
      .replace(/R\s+$/g, 'R')
      .replace(/\s{2,}/g, ' ')
      .trim();
    return next;
  }

  function sanitizeDocument(doc) {
    if (!doc || !doc.body) return;

    for (const [id, text] of Object.entries(SPECIFIC_TEXT)) {
      const el = doc.getElementById(id);
      if (el && el.textContent !== text) el.textContent = text;
    }

    const walker = doc.createTreeWalker(doc.body, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    for (const node of nodes) {
      const cleaned = cleanText(node.nodeValue);
      if (cleaned !== node.nodeValue) node.nodeValue = cleaned;
    }
  }

  function start() {
    let doc = null;
    try {
      if (window.parent && window.parent !== window) doc = window.parent.document;
    } catch (_err) {
      return;
    }
    if (!doc || !doc.body || doc.__cfMojibakeSanitizerInstalled) return;
    doc.__cfMojibakeSanitizerInstalled = true;

    const run = () => sanitizeDocument(doc);
    run();
    window.setTimeout(run, 250);
    window.setTimeout(run, 1000);
    window.setTimeout(run, 2500);

    const observer = new MutationObserver(() => run());
    observer.observe(doc.body, {
      childList: true,
      subtree: true,
      characterData: true,
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
}());
