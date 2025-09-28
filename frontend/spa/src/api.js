export async function fetchJson(path, options = {}) {
  try {
    const resp = await fetch(path, options);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return await resp.json();
  } catch (err) {
    return { error: err.message };
  }
}
