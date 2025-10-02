export function isFilledArray(v: unknown): boolean {
  return Array.isArray(v) && v.length > 0;
}
