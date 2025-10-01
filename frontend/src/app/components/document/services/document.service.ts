import { Injectable, inject } from '@angular/core';
import { map, Observable } from 'rxjs';
import { RestHttpClient } from '@core/rest-http-client/rest-http-client.service';

export interface InferDocReq {
  file: File;
  img_width: string;
  img_height: string;
}

export interface SegmentOut {
  coords: [number, number];
  value: string;
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
@Injectable({ providedIn: 'root' })
export class DocumentService {
  private readonly http = inject(RestHttpClient);

  predict(req: InferDocReq): Observable<SegmentOut[]> {
    const fd = new FormData();
    fd.append('file', req.file, req.file.name);
    fd.append('img_width', req.img_width);
    fd.append('img_height', req.img_height);

    return this.http.post<SegmentsResponse>('/ocr_segments', fd).pipe(map(req => req?.segments || []));
  }

  serverConnection(): Observable<void> {
    return this.http.get<void>('/health');
  }

  test(): Observable<void> {
    return this.http.post<void>('/train/start', {
      "mode": "train",
      "subset": "full",
      "data_root": "data/handwritten",
      "run_dir": "runs/default",
      "epochs": 40,
      "batch_size": 64,
      "monitor_metric": "cer"
    });
  }
}
