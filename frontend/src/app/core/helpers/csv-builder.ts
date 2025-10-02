// csv.ts
export function csvEscape(v: unknown, delimiter = ','): string {
  let s = String(v ?? '');
  // экранируем, если есть кавычки, перевод строки или сам разделитель
  if (s.includes('"') || s.includes('\n') || s.includes('\r') || s.includes(delimiter)) {
    s = '"' + s.replace(/"/g, '""') + '"';
  }
  return s;
}

// --- селекторы полей ---
export type FieldSelector<T> =
  | keyof T
  | string            // поддерживает вложенные пути: 'coords[0]', 'coords.1', 'meta.size.w'
  | number            // если строка — массив
  | ((row: T) => unknown);

// безопасный доступ по пути вида a.b.c или coords[0]
function getByPath(obj: any, path: string): unknown {
  if (obj == null || !path) return undefined;
  // преобразуем [0] в .0 и сплитим по точкам
  const tokens: string[] = [];
  const re = /[^.[\]]+|\[(\d+)\]/g;
  let m: RegExpExecArray | null;
  while ((m = re.exec(path)) !== null) {
    tokens.push(m[1] ?? m[0]);
  }
  let cur = obj;
  for (const t of tokens) {
    if (cur == null) return undefined;
    const key: any = /^\d+$/.test(t) ? Number(t) : t;
    cur = cur[key];
  }
  return cur;
}

function getBySelector<T>(row: T, sel: FieldSelector<T>): unknown {
  if (typeof sel === 'function') return sel(row);
  if (typeof sel === 'number')   return (row as any)?.[sel];
  if (typeof sel === 'string') {
    // прямой ключ или вложенный путь
    if (sel in (row as any)) return (row as any)[sel];
    return getByPath(row, sel);
  }
  // keyof T
  return (row as any)?.[sel as any];
}

function toLabel<T>(sel: FieldSelector<T>, i: number): string {
  if (typeof sel === 'string' || typeof sel === 'number') return String(sel);
  if (typeof sel === 'function') return `col${i + 1}`;
  return String(sel as any);
}

/**
 * Универсальный генератор CSV.
 * @param rows      данные (массив сырых значений из getRawValue())
 * @param selectors селекторы колонок: 'coords[0]' | 'strValue' | (r)=>...
 * @param labels    подписи колонок (если не заданы, берутся из селекторов)
 * @param opts      delimiter и includeHeader
 */
export function buildCsv<T>(
  rows: T[],
  selectors: FieldSelector<T>[],
  labels?: string[],
  opts?: { delimiter?: string; includeHeader?: boolean }
): string {
  const delimiter = opts?.delimiter ?? ',';
  const includeHeader = opts?.includeHeader ?? true;

  const header = labels && labels.length
    ? labels
    : selectors.map((s, i) => toLabel(s, i));

  const out: string[] = [];
  if (includeHeader) {
    out.push(header.map(h => csvEscape(h, delimiter)).join(delimiter));
  }

  for (const row of rows ?? []) {
    if (!row) continue;
    const vals = selectors.map(sel => getBySelector(row, sel));
    out.push(vals.map(v => csvEscape(v, delimiter)).join(delimiter));
  }
  return out.join('\n');
}

/**
 * Совместимая с твоей исходной сигнатура: поля — только строки (пути), шапка — labels.
 */
export function buildCsvFromFormArray<T extends object>(
  obj: T[],
  fields: string[],
  labels: string[]
): string {
  return buildCsv<T>(obj, fields, labels, { delimiter: ',', includeHeader: true });
}

/** Скачать CSV как файл */
export function downloadCsv(csv: string, filename = 'fields.csv') {
  const blob = new Blob(['\uFEFF' + csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
