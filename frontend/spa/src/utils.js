export function formatCurrency(value) {
  if (value === null || value === undefined) return '—';
  const num = typeof value === 'number' ? value : Number(value);
  if (Number.isNaN(num)) return value;
  return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(num);
}

export function formatQuantity(value) {
  if (value === null || value === undefined) return '—';
  const num = typeof value === 'number' ? value : Number(value);
  if (Number.isNaN(num)) return value;
  return new Intl.NumberFormat('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 4 }).format(num);
}

export function sortBy(list, key, direction = 'desc') {
  const sorted = [...list].sort((a, b) => {
    const av = a[key];
    const bv = b[key];
    // handle numbers
    if (typeof av === 'number' && typeof bv === 'number') return av - bv;
    // fallback to string compare
    return String(av).localeCompare(String(bv));
  });
  return direction === 'asc' ? sorted : sorted.reverse();
}
