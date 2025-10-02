/**
 * клиентские вызовы OCR API.
 */
import { Injectable, inject } from '@angular/core';
import { EMPTY, from, map, mergeMap, Observable, takeWhile } from 'rxjs';
import { RestHttpClient } from '@core/rest-http-client/rest-http-client.service';
import { HttpClient, HttpEventType, HttpHeaders } from '@angular/common/http';

export interface InferDocReq {
  file: File;
  img_width: string;
  img_height: string;
}

export interface SegmentOut {
  coords: [number, number];
  value: string;
  preview_value: string;
  wer: number;
  width: number;
  height: number;
}

export interface SegmentsResponse {
  segments: SegmentOut[];
  stats: {
    total_lines: number;
    fallback_to_trocr: number;
    conf_threshold: number;
    wer_mode: string;
  };
}

export type PredictEvent =
  | { kind: 'progress'; data: OcrProgress }
  | { kind: 'result'; data: SegmentsResponse };

export interface OcrProgress {
  step?: string;
  percent?: number;  // 0..100
  current?: number;
  total?: number;
  message?: string;
  [k: string]: any;
}

@Injectable({ providedIn: 'root' })
export class DocumentService {
  private readonly http = inject(RestHttpClient);
  private readonly httpRaw = inject(HttpClient);
  private _buffer = '';

  // Простая версия без стрима
  predict(req: InferDocReq): Observable<SegmentOut[]> {
    const fd = new FormData();
    fd.append('file', req.file, req.file.name);
    fd.append('img_width', req.img_width);
    fd.append('img_height', req.img_height);
    fd.append('stream', '1');
    return this.http.post<SegmentsResponse>('/ocr_segments', fd).pipe(map(r => r?.segments || []));
  }

  serverConnection(): Observable<void> {
    return this.http.get<void>('/health');
  }

  // Потоковая версия (SSE-like) через partialText
  predictEvents(req: InferDocReq): Observable<PredictEvent> {
    const fd = new FormData();
    fd.append('file', req.file, req.file.name);
    fd.append('img_width', req.img_width);
    fd.append('img_height', req.img_height);
    fd.append('stream', '1');

    // Низкоуровневый request: получаем HttpDownloadProgressEvent.partialText
    return this.httpRaw.request('POST', '/ocr_segments', {
      body: fd,
      headers: new HttpHeaders(), // при необходимости можно указать 'Accept': 'text/event-stream'
      observe: 'events',
      reportProgress: true,
      responseType: 'text',
    }).pipe(
      mergeMap(event => {
        if (event.type === HttpEventType.DownloadProgress) {
          const chunk = (event as any)?.partialText ?? '';
          if (!chunk) return EMPTY;
          return from(this.parseSseChunk(chunk));
        }
        if (event.type === HttpEventType.Response) {
          // финальный response тела нам не нужен — все уже пришло в чанках
          return EMPTY;
        }
        return EMPTY;
      }),
      // Завершаем поток после первого 'result', но пропускаем его наружу (inclusive)
      takeWhile(e => e.kind !== 'result', true)
    );
  }

  /**
   * Инкрементальный парсер SSE:
   *  - поддерживает накопление в this._buffer, разбиение по двойному \n\n
   *  - игнорирует "message"/keep-alive
   *  - устойчив к частичным JSON (try/catch)
   */
  private parseSseChunk(chunk: string): PredictEvent[] {
    this._buffer += chunk;
    const out: PredictEvent[] = [];
    let idx: number;

    while ((idx = this._buffer.indexOf('\n\n')) !== -1) {
      const raw = this._buffer.slice(0, idx).trim();
      this._buffer = this._buffer.slice(idx + 2);
      if (!raw) continue;

      let eventType = 'message';
      let dataStr = '';
      for (const line of raw.split('\n')) {
        if (line.startsWith('event:')) eventType = line.slice(6).trim();
        if (line.startsWith('data:')) dataStr += line.slice(5).trim();
      }
      if (!dataStr) continue;

      let payload: any;
      try {
        payload = JSON.parse(dataStr);
      } catch {
        // Частичный JSON — копим дальше
        this._buffer = dataStr + '\n\n' + this._buffer;
        continue;
      }

      if (eventType === 'progress') {
        out.push({ kind: 'progress', data: payload });
      } else if (eventType === 'result') {
        out.push({ kind: 'result', data: payload as SegmentsResponse });
      } else if (eventType === 'error') {
        throw new Error(payload?.detail || 'Server error');
      } else {
        // keep-alive / message — игнорируем
      }
    }

    return out;
  }
}
