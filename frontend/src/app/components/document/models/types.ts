/**
 * Shared types for the document components.
 */
export type OcrBackendResponse = Array<{ coords: [number, number]; value: string }>;

export interface DocumentFieldModel {
  x: number;
  y: number;
  /** Editable value (user input). */
  strValue: string;
  /** OCR suggestion (read-only hint). */
  strOcr: string;
}

export interface DocumentFormValue {
  document_name: string;
  file: string; // base64 data URL
  fields: DocumentFieldModel[];
}
