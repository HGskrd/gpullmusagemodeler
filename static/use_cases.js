/* Use-cases page: definition library load + import/export.
 * Shared wiring (tab id, headers, toasts, live sliders, HTMX listeners) lives in app.js. */
function replaceLibrary(html) {
  const target = document.getElementById('useCaseLibrary');
  if (!target) return;
  target.innerHTML = html;
  if (window.htmx) htmx.process(target);
  bindLiveSliders(target);
}

function exportUseCaseLibrary() {
  fetch('/use-cases/export', { headers: requestHeaders() })
    .then(async r => {
      if (r.ok) return r.blob();
      let message = await r.text();
      try {
        const data = JSON.parse(message);
        if (data && data.error) message = data.error;
      } catch {}
      throw new Error(message || `Export failed (${r.status})`);
    })
    .then(blob => {
      const objectUrl = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = objectUrl;
      a.download = 'use-case-library.json';
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(objectUrl);
    })
    .catch(err => showToast(err.message));
}

function importUseCaseLibrary(input) {
  const file = input && input.files && input.files[0];
  if (!file) return;
  file.text()
    .then(text => postPartial('/use-cases/import', new URLSearchParams({ json: text }).toString()))
    .then(replaceLibrary)
    .catch(err => showToast(err.message))
    .finally(() => { input.value = ''; });
}

document.addEventListener('DOMContentLoaded', () => {
  fetch('/use-cases/library', { headers: requestHeaders({ 'HX-Request': 'true' }) })
    .then(r => r.text())
    .then(replaceLibrary)
    .catch(console.error);
});
