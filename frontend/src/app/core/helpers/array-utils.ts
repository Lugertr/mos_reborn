export function isFilledArray(v: unknown): boolean {
  return Array.isArray(v) && v.length > 0;
}

export function isArraysEqual<T extends number | string>(ar1: T[] | null, ar2: T[] | null): boolean {
  if (ar1 === ar2) return true;
  if (!Array.isArray(ar1) || !Array.isArray(ar2)) return false;
  if (ar1.length !== ar2.length) return false;
  ar1.sort();
  ar2.sort();
  for (let i = 0; i < ar1.length; i++) {
    if (ar1[i] !== ar2[i]) return false;
  }
  return true;
}

export function chunkArray<T>(arr: T[], chunkSize: number): T[][] {
  const result: T[][] = [];
  for (let i = 0; i < arr.length; i += chunkSize) {
    result.push(arr.slice(i, i + chunkSize));
  }
  return result;
}

/** Fork of lodash arrayIncludesWith */
export function arrayIncludesWith<T>(arr: T[], value: T, comparator: (a: T, b: T) => boolean): boolean {
  if (arr.length === 0) return false;

  for (const item of arr) {
    if (comparator(value, item)) return true;
  }

  return false;
}

export function uniqWith<T>(items: T[], equalBy: (a: T, b: T) => boolean): T[] {
  const result: T[] = [];
  if (items.length === 0) return result;

  let val: T = items[0];
  result.push(val);

  for (let i = 1; i < items.length; i++) {
    val = items[i];
    if (!arrayIncludesWith(result, val, equalBy)) {
      result.push(val);
    }
  }
  return result;
}
