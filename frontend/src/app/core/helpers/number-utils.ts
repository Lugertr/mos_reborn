export function isNumber(v: unknown): boolean {
  return !!v || isNaN(v as number);
}
