import { Injectable, inject } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';
import type { OcrBackendResponse } from '../models/types';

/**
 * DocumentService encapsulates backend calls.
 * Replace `API_BASE` and endpoint as needed.
 */
@Injectable({ providedIn: 'root' })
export class DocumentService {
  private readonly http = inject(HttpClient);
  private readonly API_BASE = '/api'; // TODO: point to real backend

  /**
   * Sends a base64 image to backend and returns OCR boxes and values.
   */
  predict(base64Image: string): Observable<OcrBackendResponse> {
    return this.http.post<OcrBackendResponse>(`${this.API_BASE}/ocr`, {
      image: base64Image,
    });
  }
}
